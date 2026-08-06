# Fase 1 — Como a política de locomoção (walk + run) do G1 funciona

Este documento explica em detalhe o que está acontecendo no treino da política unificada
de caminhada+corrida do G1 no MimicKit — o que é sorteado, como o reward é montado, como
o discriminador AMP funciona — **e** toda a infraestrutura que precisou ser construída/
corrigida no caminho: bugs no MimicKit em si, migração de assets, monitoramento, ablação,
e o testador de sim2sim em MuJoCo (ainda em depuração no momento em que isso foi escrito).
Todos os valores citados são os que estão de fato nos arquivos de config em
`data/envs/amp_steering_g1_env.yaml`, `data/datasets/dataset_g1_locomotion.yaml` e
`data/agents/amp_task_g1_agent.yaml`.

## Linha do tempo resumida

1. Setup do MimicKit no `env_isaaclab` (conda/uv, versões).
2. Avaliação de dois papers (Olaf — impact reduction; Grandia et al. — confirmado que
   **não** é um paper de navegação) e decisão de escopo: Fase 1 (locomoção) antes de
   Fase 2 (switcher de políticas) e Fase 3 (navegação).
3. Port do asset do G1 + retargeting provisório via CopyCat/GMR → **substituído** pelo
   pacote oficial `MimicKit_Data` (assets/motions/checkpoints prontos).
4. Bug crítico no `isaac_lab_engine.py` (`link_parent_indices` não existe mais na API do
   PhysX instalada) — corrigido reconstruindo a hierarquia via reparse do MJCF.
5. Cadeia de bugs no pipeline de vídeo headless (render gate, câmera, draw_interface,
   timing) — todos corrigidos.
6. Implementação do reward de impact-reduction (seção 6).
7. Treino real no Isaac Lab + wandb (projeto `MimicKit`), com um estudo de ablação
   (impact-reduction ligado vs. desligado).
8. Extensão pedida: unificar STAND+WALK+RUN numa política só (em andamento — precisa de
   um `g1_stand.pkl`, que ainda não existe pronto).
9. Testador de sim2sim em MuJoCo puro (headless por terminal + janela ao vivo via X11) —
   várias camadas de bug encontradas e corrigidas (PD duplicado, integrador, cenário);
   **ainda não validado como estável** no momento em que este documento foi escrito.
10. Investigação de reuso do `motion_tracking_controller` (pipeline ONNX/ROS2 do
    BeyondMimic) — **veredito: não é reutilizável como está**, contrato de observação/ação
    fundamentalmente diferente (seção 14).

---

## 1. Uma política, dois estilos de movimento

Não existem duas políticas (uma pra andar, uma pra correr). Existe **uma rede** treinada
com o método `task_steering` (AMP + comando de tarefa), que recebe como parte da sua
observação um vetor de comando de velocidade — direção 2D + magnitude (`tar_speed`) — e
tem que produzir um movimento que (a) alcance essa velocidade e (b) pareça humano. Andar e
correr não são dois "modos" chaveados: são dois pontos num espaço contínuo de
comportamentos que a mesma rede aprende a interpolar.

Isso é possível porque o AMP não exige que o robô reproduza um clipe específico
frame-a-frame (ver seção 5) — ele só precisa "parecer real" localmente, então a rede é
livre para produzir qualquer velocidade/estilo intermediário que ainda pareça natural.

---

## 2. O que é sorteado, e quando

Em `mimickit/envs/task_steering_env.py`, a função `_reset_tar` roda toda vez que um
ambiente é resetado (personagem caiu, ou trocou de alvo):

```python
rand_theta  = uniforme(-pi, pi)                                  # direção alvo
tar_dir     = [cos(rand_theta), sin(rand_theta)]                 # vetor direção 2D
tar_speed   = uniforme(tar_speed_min, tar_speed_max)              # magnitude
face_dir    = tar_dir   # porque rand_face_dir: False no nosso config
```

- **`tar_speed_min: 0.5`, `tar_speed_max: 3.5`** (m/s) — sorteado uniformemente em cada
  reset de alvo. Cobre desde "andar devagar" (0.5, abaixo da velocidade medida do clipe de
  walk) até um pouco acima do clipe de run (3.5 vs. ~3.4 medido).
- **`tar_dir`** — direção sorteada em qualquer ângulo (`rand_tar_dir: True`), 360°
  completos. O robô tem que aprender a virar, não só andar reto.
- **`rand_face_dir: False`** — o personagem sempre encara a direção pra onde está indo
  (não anda de lado/de costas nesta fase).
- **Frequência de resorteio**: `tar_change_time_min: 4.0` / `tar_change_time_max: 7.0`
  segundos — a cada 4–7s (sorteado também), o ambiente troca o alvo de
  velocidade/direção, então dentro de um único episódio (`episode_length: 10.0`s) o robô
  pode receber 1–2 comandos diferentes.
- Cada um dos `num_envs` ambientes paralelos (4096 no nosso treino) sorteia isso de forma
  **independente** — a cada instante, milhares de combinações diferentes de
  velocidade/direção estão sendo exploradas simultaneamente.

---

## 3. Os clipes de referência

`data/datasets/dataset_g1_locomotion.yaml`:

```yaml
motions:
  - file: "data/motions/g1/g1_walk.pkl"
    weight: 1.0
  - file: "data/motions/g1/g1_run.pkl"
    weight: 1.0
```

Ambos vêm do pacote oficial do MimicKit (`MimicKit_Data`), não foram criados por nós.
Medimos a velocidade real (root speed) desses clipes diretamente:

| Clipe | FPS | Frames | Velocidade média medida |
|---|---|---|---|
| `g1_walk.pkl` | 120 | 125 | ~1.0 m/s (bem estável) |
| `g1_run.pkl` | 30 | 25 | ~2.8–3.4 m/s |

`weight: 1.0` em ambos significa que o `motion_lib` (que sampleia clipes pra montar as
observações do discriminador, ver seção 5) sorteia entre os dois com a mesma
probabilidade — não há viés pra um ou outro.

**Gap conhecido**: não existe um clipe de caminhada "pra frente" em velocidade
intermediária nem clipes de estilos alternativos (andar de lado, mancando, etc.) — só
esses dois. A política vai ter que **extrapolar** o intervalo entre 1.0 e 2.8 m/s sozinha,
apoiada só no reward de velocidade + no discriminador. Isso deve funcionar (é exatamente
pra isso que o AMP serve), mas é um ponto pra observar quando for avaliar o resultado.

---

## 4. O reward, termo por termo

O reward total tem duas partes, calculadas em lugares diferentes e combinadas no agente:

```
r_total = task_reward_weight · r_task + disc_reward_weight · r_AMP
        = 0.5 · r_task + 0.5 · r_AMP        (amp_task_g1_agent.yaml)
```

### 4.1 `r_task` — calculado em `task_steering_env.py::_update_reward`

```python
r_task = reward_steering_tar_w  · r_velocidade
       + reward_steering_face_w · r_direção
       + reward_impact_w        · r_impacto
```

| Termo | Peso | Fórmula | O que faz |
|---|---|---|---|
| `r_velocidade` | 0.7 | `exp(-vel_err_scale · erro²)`, `vel_err_scale=0.5` | Recompensa a velocidade do torso projetada na direção alvo ficar perto de `tar_speed`. Zerado se o robô estiver indo na direção errada. |
| `r_direção` | 0.3 | `clamp_min(dot(tar_dir, facing_dir), 0)` | Recompensa o torso estar de frente pro sentido do movimento. |
| `r_impacto` | 2.5e-3 | ver seção 6 | Reward de impact-reduction (Olaf). Peso bem menor que os outros — é um "tempero" de qualidade, não o objetivo principal. |

**Nota importante**: o `amp_steering_g1_env.yaml` ainda tem campos `reward_pose_w`,
`reward_vel_w`, `reward_root_pose_w`, `reward_key_pos_w` etc. (herdados do template que
clonamos). O `TaskSteeringEnv` **não lê esses campos** — quem faz o trabalho de "parecer
humano" é só o discriminador AMP (seção 5), não um tracking de pose manual. Esses campos
estão no arquivo mas são inertes; podem ser removidos sem efeito.

### 4.2 `r_AMP` — calculado em `amp_agent.py::_calc_disc_rewards`

```python
prob  = sigmoid(discriminador(disc_obs))     # probabilidade do discriminador achar "real"
r_AMP = -log(max(1 - prob, 0.0001)) · disc_reward_scale     # disc_reward_scale = 2
```

Isso é a forma clássica de reward do AMP (non-saturating GAN loss): quanto mais o
discriminador acha que o movimento do robô é "real" (prob → 1), maior o reward — ele
cresce sem limite conforme `prob` se aproxima de 1, incentivando a política a continuar
melhorando o estilo mesmo depois de já enganar o discriminador na maior parte do tempo.

---

## 5. O discriminador — como funciona de verdade

Diferente de motion tracking clássico (DeepMimic), o discriminador não compara o robô
contra um frame específico do clipe. Ele é uma rede neural binária, treinada
adversarialmente (como a parte "discriminadora" de uma GAN), que olha **janelas de 10
frames consecutivos** (`num_disc_obs_steps: 10`) e tenta classificar: "isso é movimento
real (veio do dataset) ou movimento do robô simulado?"

### O que entra na observação do discriminador (`disc_obs`)

Construído em `amp_env.py::_compute_disc_obs_demo`, por janela de 10 frames:
- posição e rotação da raiz (torso), **relativas ao último frame da janela** — ou seja,
  invariante a posição/orientação global. O discriminador não sabe (nem se importa) onde
  no mundo o robô está ou pra que lado está virado; só olha a dinâmica local do movimento.
- velocidade linear e angular da raiz
- rotação de cada junta (`joint_rot`)
- velocidade de cada junta (`dof_vel`)
- posição dos "key bodies" (`left_ankle_roll_link`, `right_ankle_roll_link`, `head_link`,
  `left_wrist_yaw_link`, `right_wrist_yaw_link`)

### Como ele é treinado

Em cada iteração de treino, o discriminador vê dois grupos de amostras:
1. **"Real"**: janelas de 10 frames tiradas **aleatoriamente** de dentro de
   `g1_walk.pkl`/`g1_run.pkl` (não sincronizadas com nenhuma fase — pode pegar qualquer
   trecho de qualquer um dos dois clipes, com probabilidade igual, per `weight: 1.0` em
   ambos).
2. **"Fake"**: janelas de 10 frames tiradas do rollout do próprio robô em simulação (mais
   um buffer de replay de até 200.000 amostras passadas, `disc_buffer_size: 200000`, pra
   não esquecer comportamentos antigos).

O discriminador é otimizado pra separar essas duas classes; a política (ator) é
otimizada simultaneamente pra **maximizar a confusão do discriminador** — ou seja, gerar
movimento que ele não consiga distinguir do real. Esse é o mecanismo clássico de GAN
aplicado a controle de movimento.

### Por que isso é diferente (e melhor pra esse caso) que tracking direto

| | DeepMimic (tracking direto) | AMP (discriminador) |
|---|---|---|
| O que compara | Pose exata num instante `t` específico do clipe | "Essa dinâmica de 10 frames parece real?" |
| Precisa de fase (`φ_t`) | Sim — saber exatamente onde está no clipe | Não |
| Funciona pra velocidade fora do dataset | Não (não existe frame de referência) | Sim — só precisa parecer natural, a velocidade exata vem do `r_task` |
| Robustez a datasets pequenos/variados | Baixa (cada clipe precisa de tuning) | Alta |
| Precisão pra movimento específico (cartwheel) | Alta | Baixa (por isso usamos DeepMimic pras episódicas) |

---

## 6. O reward de impact-reduction (Olaf, arXiv:2512.16705)

Implementado genericamente em `mimickit/envs/char_env.py` (reutilizável por qualquer
personagem/ambiente, não é específico do G1), ligado no `task_steering_env.py`.

### Fórmula

```
r_impacto = - Σ_{i ∈ {pé esquerdo, pé direito}}  min(Δv_{i,z}² , Δv_max²)
```

- `Δv_{i,z}` = variação de velocidade **vertical** (eixo Z) do pé `i` entre dois steps
  consecutivos de simulação.
- `Δv_max²` = saturação — no nosso config, `impact_vel_max: 3.5` m/s (não é o número
  específico do paper Olaf, que não estava claramente disponível pra nós; usamos o mesmo
  valor do `tar_speed_max` do config por consistência, já que é a ordem de grandeza de
  velocidade relevante pro robô nesta tarefa).
- `impact_bodies: ["left_ankle_roll_link", "right_ankle_roll_link"]` — só os pés contam.

### Implementação (mecânica exata)

1. **Antes de cada physics step** (`char_env.py::_pre_physics_step`), grava a velocidade Z
   atual dos pés em `self._prev_impact_body_vel_z` (isso é o "antes").
2. **Depois do physics step**, `_compute_impact_reduction_reward()` lê a velocidade Z
   atual dos pés (o "depois"), calcula `Δv = atual - anterior`, eleva ao quadrado,
   satura em `impact_vel_max²`, soma os dois pés e nega.
3. Esse valor é multiplicado por `reward_impact_w: 2.5e-3` e somado ao `r_task`.

### Por que isso funciona

Um impacto forte do pé no chão (pisada "batida") gera uma mudança brusca e grande na
velocidade vertical do pé no instante do contato (de uma velocidade descendente alta pra
zero quase instantaneamente). Penalizar isso empurra a política a **desacelerar o pé
verticalmente antes de tocar o chão** — pousar suave em vez de bater. No robô físico do
paper Olaf, esse mecanismo produziu 13.5 dB de redução no ruído do passo. A saturação
(`min(..., Δv_max²)`) existe pra que picos de resolução de contato do próprio motor de
física (que podem ser artificialmente grandes por um instante) não dominem o gradiente e
desestabilizem o treino do critic.

---

## 7. Juntando tudo — o que a rede recebe e produz

**Observação da política** (`compute_char_obs` em `char_env.py` + `compute_steering_observations`
em `task_steering_env.py`): altura da raiz, orientação, velocidade linear/angular do
torso, rotação e velocidade de cada uma das 29 juntas, posição dos key bodies, **mais** o
comando de tarefa: direção alvo e velocidade alvo em coordenadas locais do personagem, e a
direção de "encarar".

**Ação da política**: setpoints de posição para as 29 juntas (controlador PD por baixo).

**Rede**: duas camadas fully-connected de 1024 unidades (`fc_2layers_1024units`) para
ator, crítico e discriminador — três redes separadas, mesma arquitetura de tamanho.

**Treino**: PPO (`ppo_clip_ratio: 0.2`), `steps_per_iter: 32` (samples coletados por
ambiente por iteração), `actor_epochs: 5` / `critic_epochs: 2` / `disc_epochs: 2` por
iteração, `discount: 0.99`, `td_lambda: 0.95`. A cada `iters_per_output: 200` iterações,
roda uma avaliação (`test_episodes: 32`) e salva checkpoint em `output/g1_locomotion/`.

---

## 8. Checklist de avaliação (quando o checkpoint estiver pronto)

- [ ] `tar_speed` baixo (~0.5-1.0) produz uma caminhada estável, sem arrastar os pés
- [ ] `tar_speed` alto (~2.5-3.5) produz uma corrida reconhecível, não um "shuffling" rápido
- [ ] Transição de velocidade dentro do range não sorteado nos clipes (ex: 1.8 m/s) ainda
      parece natural — é o teste real de que o AMP está generalizando, não só memorizando
      os dois clipes
- [x] Comparar o perfil de `Δv_z` do pé no contato com/sem o termo de impact-reduction —
      feito como estudo de ablação de verdade (seção 12), não só um flag local
- [ ] Robô consegue virar (seguir `tar_dir` em ângulos variados), não só andar reto

---

## 9. Infraestrutura — bugs corrigidos no próprio MimicKit

Antes de qualquer treino rodar, vários bugs de compatibilidade do MimicKit com a versão
instalada de Isaac Sim/Isaac Lab (4.5) precisaram ser corrigidos. Nenhum desses é
específico do G1 — afetam qualquer personagem/tarefa rodando no `isaac_lab_engine.py`.

### 9.1 `link_parent_indices` não existe mais na API do PhysX

`isaac_lab_engine.py::_build_body_order` dependia de `meta_data.link_parent_indices`
pra reconstruir a hierarquia de corpos (necessário pra mapear a ordem "do simulador" pra
ordem "comum"/cinemática). Essa versão da API de tensores do PhysX **não expõe mais isso**
— nem em `shared_metatype`, nem em `root_physx_view` (confirmado via diagnóstico
instrumentado: nem os `link_paths` ajudam, porque são caminhos USD *irmãos* dentro do
prim do robô, não aninhados por parentesco).

**Fix**: `_derive_link_parent_indices_from_mjcf()` — reparseia o `.xml` de origem usando
o `MJCFCharModel` que o próprio MimicKit já usa pra cinemática, e reconstrói o dicionário
de parentesco por nome de body, comparando com os `link_names` que o PhysX de fato retorna.

### 9.2 Pipeline de gravação de vídeo headless — cadeia de 5 bugs

Pra conseguir `--mode test --visualize false --video true` funcionando (gravar vídeo sem
precisar de display), apareceram bugs em cascata, cada um revelando o próximo:

1. `run.py`'s `test()` nunca chamava o logger — vídeo era capturado mas nunca salvo.
   Fix: `save_test_video()` chama `env.record_diagnostics()` e salva o `.mp4` direto.
2. `sim_env.py::step()` só chamava `_render()` (onde o frame é capturado)
   `if self._visualize` — nunca com só `record_video=True`. Fix: `if visualize or
   record_video`.
3. `view_motion_env.py`/`env_builder.py` simplesmente não repassavam `record_video` pro
   `ViewMotionEnv` (todos os outros tipos de env repassavam corretamente). Fix: adicionado.
4. `isaac_lab_engine.py`: câmera, `draw_interface` e `_play_mode` só eram construídos
   `if visualize` — quebrava com `AttributeError` assim que `record_video`-only tentava
   renderizar. Fix: construir também quando só `record_video=True` (exceto
   `_draw_interface`, que depende de um módulo Python que não existe nessa instalação de
   Isaac Sim — `render()` agora só chama `clear_lines()` se ele existir).
5. `engine.py`'s `_prev_frame_time` (usado só pra throttle de FPS) também só era
   inicializado `if visualize`. Fix: inicializa sempre.

Também corrigimos `WandbLogger`: o nome do projeto estava fixo em `"mimickit"` (trocado
pra `"MimicKit"`), e o nome da run sempre saía `"log"` (nome fixo do arquivo de log,
independente do treino) — trocado pra usar o nome da pasta `out_dir`, que é único por
treino.

---

## 10. Migração pros assets oficiais do MimicKit

Inicialmente portamos o `g1.xml` do `whole_body_tracking` (BeyondMimic) e retargetamos
clipes via CopyCat/GMR como solução provisória. Depois veio à tona que o MimicKit tem seu
próprio pacote de assets oficial (`MimicKit_Data`, baixável via link no README) —
**migramos pra ele**, o que trouxe:

- `data/assets/g1/g1.xml` oficial — 29 DOF, com `head_link` de verdade (o que tínhamos
  portado não tinha pescoço articulado; corrigimos as configs que dependiam disso e
  depois revertemos quando trocamos pro oficial).
- `g1_walk.pkl`/`g1_run.pkl` oficiais (usados no dataset, seção 3) — muito melhores que os
  clipes provisórios do CopyCat (que não tinham um walk "pra frente" de verdade).
- 5 clipes acrobáticos prontos (`double_kong`, `kick_combo`, `spinkick`, `speed_vault`,
  `cartwheel`) e **4 checkpoints já treinados** (`deepmimic_g1_double_kong_model.pt`,
  `deepmimic_g1_spinkick_model.pt`, `lcp_g1_walk_model.pt`, `add_g1_run_model.pt`) — fora
  de escopo da Fase 1 (que é só walk+run unificado), mas relevantes pra Fase 2 (políticas
  episódicas via switcher).

---

## 11. Monitoramento — wandb

Treino roda com `--logger wandb`, projeto **`MimicKit`**, nome de run = nome da pasta
`out_dir` (seção 9.2). Vídeo periódico funciona porque o MimicKit já tem esse mecanismo
embutido: a cada `iters_per_output` iterações, `train_model()` roda uma avaliação em modo
TEST (que aciona a gravação) e sobe o resultado — não precisamos construir nada novo pra
isso, só habilitar com `--video true` e garantir que o pipeline da seção 9.2 funcionasse.

`iters_per_output` foi ajustado de `200` pra `50` especificamente pra ter feedback visual
mais frequente durante essa fase de validação (custo: mais overhead de avaliação/vídeo
durante o treino; pode voltar pra um valor mais alto depois que o comportamento estiver
validado).

---

## 12. Estudo de ablação — impact-reduction

Depois de observar que o movimento parecia menos natural com o termo de impact-reduction
ligado, criamos um par de runs pra comparar de verdade, mudando só uma coisa entre elas:

- **`g1_locomotion_impact_reduction`** (`output/g1_locomotion/`) — `reward_impact_w: 2.5e-3`
  (config original, `amp_steering_g1_env.yaml`).
- **`g1_locomotion_no_impact`** (`output/g1_locomotion_no_impact/`) —
  `reward_impact_w: 0.0` (`amp_steering_g1_no_impact_env.yaml`), tudo o resto idêntico.

Comparação feita via wandb (mesmo projeto, duas runs lado a lado) — `Test_Return`,
`Train_Return` e o vídeo periódico de cada uma.

---

## 13. Extensão pedida: STAND unificado (em andamento)

Pedido: além de walk↔run, a mesma política devia saber sair de parado (`STAND → WALK`,
`STAND → RUN`) e voltar a parar depois de correr — tudo numa política só, sem depender do
switcher da Fase 2 pra isso.

### Por que isso não acontece hoje

Confirmado no código (`deepmimic_env.py::_reset_char` → `_reset_ref_motion` →
`_ref_state_init`): com `rand_reset: True`, todo reset de episódio já inicializa o
personagem **no meio de um clipe real** (Reference State Initialization, RSI) — nunca a
partir de uma pose parada. Somado a `tar_speed_min: 0.5` (nunca pede velocidade zero) e a
ausência de qualquer clipe de "ficar parado" no dataset, a política nunca teve motivo pra
aprender esse comportamento.

### O que falta

1. **Um `g1_stand.pkl`** — não existe pronto em lugar nenhum do workspace (nem pro G1, nem
   pro humanoid genérico do próprio MimicKit). Tentamos:
   - Sintetizar um estático (`tools/gen_stand_motion.py`, segurando a pose de
     `init_pose` por 2s) — funciona tecnicamente, mas a pose (cotovelos a 90°, mãos perto
     do quadril) não é uma postura de "em pé relaxado" de verdade — é só a config de
     segurança genérica usada pra evitar auto-colisão no reset, não algo pensado pra
     imitar um "idle" natural.
   - Retargetar via GMR — melhor qualidade esperada, mas precisa de uma fonte de mocap de
     "ficar parado"/idle. Achamos pistas (`CopyCat/retarget/GENMO/outputs/anchors/
     stand_still_30f_s050000`, `stand_still_60f_s050000`, `GMR/output/pkl/
     booster_t1_stand_still.pkl` — esse último é de outro robô, o Booster T1) — **ainda
     não confirmamos se dá pra usar isso pro G1**, ficou pendente.
2. Depois de ter o clipe: adicionar ao `dataset_g1_locomotion.yaml` e abrir
   `tar_speed_min` pra incluir perto de zero.

---

## 14. Testador de sim2sim em MuJoCo (`tools/sim2sim_mujoco/`)

Objetivo: validar a política treinada rodando de verdade fora do Isaac Lab, num motor de
física diferente (MuJoCo puro), controlado via teclado (WASD-like) em vez de joystick —
sem depender de ONNX/ROS2.

### 14.1 Por que não dá pra reusar o `motion_tracking_controller`

Investigamos se dava pra exportar nossa política pro pipeline C++/ROS2 que o
`motion_tracking_controller` (companion do BeyondMimic) já tem pronto, com MuJoCo +
joystick + deploy real prontos. **Veredito: não, incompatibilidade estrutural**, não é
questão de escrever um script de export:

- O ONNX que esse controlador espera **embute o clipe de referência inteiro** como
  tensores constantes, indexados por um contador `time_step` — literalmente "toca essa
  coreografia específica", sem noção nenhuma de comando de velocidade ao vivo em lugar
  nenhum do código C++.
- Observação e ação usam convenções diferentes das nossas (ângulo escalar por junta vs.
  nossa representação 6D por rotação; ação como delta+offset da pose default vs. nossa
  ação como alvo absoluto centrado em zero).
- MimicKit não tem exportador ONNX próprio — teria que ser escrito do zero, e mesmo assim
  precisaria de um controlador C++ novo (não o existente) pra consumir comando de
  velocidade ao vivo.

Por isso: testador próprio, direto em Python/PyTorch, sem ONNX.

### 14.2 Arquitetura

- `keyboard_test.py` — versão headless (SSH puro): lê teclado via `termios`/`tty` (sem
  precisar de janela), imprime telemetria no terminal, salva vídeo periodicamente.
- `keyboard_test_viewer.py` — versão com janela ao vivo (`mujoco.viewer`), precisa de
  X11 forwarding (`ssh -X`). Reaproveita as mesmas classes/funções do script headless via
  import, não duplica lógica de inferência.
- `preview_pose.py` — utilitário pra checar uma pose estática (usado pra validar o
  `STAND_POSE` antes de virar clipe, seção 13).
- Carrega o checkpoint `.pt` **direto em PyTorch** (sem ONNX): extrai só o que precisa pra
  inferência do `state_dict` do agente — `_obs_norm`/`_a_norm` (média/desvio) e os pesos
  do `_actor_layers`/`_action_dist._mean_net`, ignorando crítico e discriminador (não são
  necessários em inferência).
- Observação construída chamando as **mesmas funções** que o treino usa
  (`char_env.compute_char_obs`, `task_steering_env.compute_steering_observations`,
  `MJCFCharModel.dof_to_rot`/`forward_kinematics`) em vez de reimplementar a matemática —
  reduz risco de bug de convenção.
- Controle: W/X = acelera/desacelera na direção atual (emula eixo vertical do analógico
  esquerdo), A/D = taxa de giro contínua do heading (emula eixo horizontal), S = para —
  **não** é D-pad de 4 direções fixas, é o mesmo modelo "unicycle" (throttle + steering)
  que a política foi treinada pra seguir.

### 14.3 Bugs encontrados e corrigidos (nessa ordem)

1. **Framebuffer offscreen pequeno demais** (`preview_pose.py` pedia 960×720, padrão do
   MuJoCo é 640×480) — fix: `model.vis.global_.offwidth/offheight` setado em runtime.
2. **`MUJOCO_GL` não configurado** — sem display real, o backend padrão (GLFW) falha.
   Fix: `MUJOCO_GL=egl` (acelerado por GPU) pro modo headless.
3. **Pose "torta" na primeira imagem de preview** — não era bug de rotação (o
   `exp_map_to_quat` já trata ângulo zero corretamente, conferimos o código). Era falta de
   plano de chão/referência visual na cena + câmera num ângulo oblíquo sem nenhum ponto de
   referência de "vertical" — qualquer pose reta podia parecer torta assim. Fix: injeção
   de chão + câmera nivelada de frente.
4. **Caminho de mesh duplicado** (`meshes/meshes/...`) ao carregar de um XML temporário
   fora da pasta original — `meshdir` tem que apontar pra pasta do `g1.xml`, não pra
   subpasta `meshes/` (os `file=""` dos meshes já incluem esse prefixo).
5. **Cenário não era o esperado** — pedido explícito de bater com o visual usado no
   `motion_tracking_controller`. Como esse pacote (`unitree_description`, via ROS2) não
   está instalado nessa máquina, usamos como referência o `scene.xml` padrão do G1 que já
   existe em outro projeto do workspace (`twist3/TWIST2/assets/g1/scene.xml`) — chão
   xadrez com textura, skybox liso, headlight — é a convenção usada na maioria dos repos
   de G1/MuJoCo por aqui, bem provável que seja o mesmo visual.
6. **"Explosão" da simulação** (câmera/robô disparando pra longe) — causa real: **PD
   duplicado**. O `<joint stiffness=".." damping=".."/>` do MJCF já é aplicado
   automaticamente pelo MuJoCo como mola/amortecedor passivo a cada step, independente de
   `data.ctrl` — o cálculo manual de torque que eu fazia (usando os mesmos valores, via
   atuador `<motor>`) contava a rigidez **duas vezes**. Fix: usar o mecanismo nativo do
   MuJoCo (`model.qpos_spring`, o alvo da mola nativa) em vez de recalcular torque na mão.
7. **Timestep de física errado** — `g1.xml` não declara `<option timestep>`, então caía no
   padrão do MuJoCo (0.002s / 500Hz) em vez de bater com `sim_freq: 120` do
   `isaac_lab_engine.yaml` (1/120s). Fix: injetado explicitamente.
8. **Robô atravessando o chão** — não era falta de colisão (conferimos: os geoms de
   colisão do G1 usam `contype`/`conaffinity` padrão, deviam colidir normalmente com o
   plano injetado). Junto com o integrador Euler explícito padrão do MuJoCo, a dinâmica
   ainda instabilizava o bastante pra "atravessar" antes do solver corrigir. Fix:
   integrador trocado pra `implicitfast` (mais robusto pra esse tipo de mola/amortecedor
   de junta).

### 14.4 Status atual — ainda em aberto

Com todos os fixes acima, o robô já **inicializa em pé** (usando um frame real do
`g1_walk.pkl` como pose inicial, via `--init_pose walk` — isolando que não é a pose
inicial sintética o problema) e **não atravessa mais o chão nem explode**, mas ainda
**colapsa** pra uma pose sentada/encolhida pouco depois de começar, de forma consistente
(mesma pose final nas duas tentativas — não é queda aleatória).

Hipótese em investigação: os valores de `stiffness`/`damping` do MJCF **são** os mesmos
que o Isaac Lab usa de verdade (confirmado no código —
`ImplicitActuatorCfg(stiffness=None, damping=None)` pro `control_mode: pos` significa
"herda do asset importado", ou seja, os mesmos números do XML) — mas o Isaac Lab usa o
atuador implícito nativo do **PhysX** pra isso, enquanto o MuJoCo trata como
mola-amortecedor genérica. Mesmos números declarados não garantem o mesmo comportamento
efetivo de sustentação de peso entre motores de física diferentes — é um gap conhecido de
sim2sim entre engines. Adicionamos `--pd_gain_scale` como parâmetro de diagnóstico
(multiplica `jnt_stiffness`/`dof_damping`) pra testar essa hipótese empiricamente antes de
decidir o próximo passo — **resultado desse teste ainda não confirmado** no momento em que
este documento foi escrito.
