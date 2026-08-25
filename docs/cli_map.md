# uav-swarm-sim — konfigų ir CLI žemėlapis

*Patikrinta prieš commit 493e8d2 (2026-08). Naujesni pakeitimai gali būti neatspindėti.*
*Šiame dokumente minėti `config/study01_demand_newrth.yaml` ir `config/shape_sweep_newrth.yaml` yra IŠIMTI (C2/C3 užšaldytos RTH konfigūracijos) — pilnas tekstas: `git show pre-reset-archive:<kelias>`; kontekstas: `docs/archive/PROJECT_HISTORY.md` §8.*

Nuoroda: kur laikomi konfigai, kas juose yra, ir ką galima perrašyti per CLI.
Visi skriptai paleidžiami per `python -m uav_swarm_sim.experiments.<vardas>` iš repo
šaknies (venv aktyvus). Variklis determinuotas: `(config, master_seed, replication,
algo, planner)` → bitas-į-bitą identiškas rezultatas.

> Palaikymas: šis dokumentas atspindi kodą būsenoje, kai buvo rašytas. Jei faktas čia
> prieštarauja kodui — **kodas laimi**; perskaityk šaltinį ir pažymėk drift'ą.
> Schema + loaderis: `src/uav_swarm_sim/infrastructure/config.py`.

---

## 1. Kur laikomi konfigai

Viskas gyvena `config/` kataloge.

| Failas | Paskirtis |
|---|---|
| `config/default.yaml` | **Vienintelis tikrasis šaltinis** — 1:1 atspindi dataclass schemą, λ=0.3 (pakeltas MC), MULTIROTOR |
| `config/djimatrice4e.yaml` | DJI Matrice 4E platforma (99.5 Wh, swath 132 m, kvadratinės kliūtys). Scale sweep'ų default |
| `config/shape_sweep_newrth.yaml` | Shape sweep su nauju RTH (energy_map) |
| `config/study01_demand.yaml` / `study01_demand_newrth.yaml` | STUDY-01 demand; `run_rth_ab.py` default |
| `config/scenarios/smoke.yaml` | Minimalus smoke (n=3, mc.n_max=6) — tik greitam testui |

**Mechanika:** `load_config(path, overrides)` skaito **VIENĄ** YAML — nėra merge/paveldėjimo
tarp failų. Kiekvienas config'as savarankiškas; neprivalomi blokai (`telemetry`, `coverage`,
`rth.energy_map`, `safety.obstacle_recovery`…) krenta į dataclass default'us, jei jų nėra faile.
`overrides` = dotted-key dict (pvz. `{"fleet.n_drones": 30}`), taikomas per `_deep_set`
**prieš** `config_hash` skaičiavimą (provenance). Vienetai YAML'e: baterija Wh, kai kurie
kampai laipsniais; loaderis viską verčia į SI (Wh→J, deg→rad).

---

## 2. Kas konfiguose yra — pagal blokus

### a) Sklypas — `env:` + `layers:`
| Raktas | Default | Reikšmė |
|---|---|---|
| `env.geojson_path` | example_area.geojson | Survey poligono forma |
| `env.coverage_altitude_m` | 100.0 | Padengimo aukštis |
| `env.clearance_buffer_m` | 5.0 | Reguliacinis buferis apie kliūtis |
| `layers.altitudes_m` | null | null = 1 sluoksnis (2D net); sąrašas → 2.5D stackas |
| `layers.assignment_policy` | single | single / area_balanced / battery_tiered |
| `coverage.ferry_free_space` / `operating_area` / `operating_margin_m` | false / convex_hull / 50 | Skrydžio zona ≠ survey poligonas (aplinkkeliai) |
| `coverage.transit_free_space` | false | Kliūtis-suvokiantis S1 transit maršrutas (FIX-B1) |

### b) Dronas — `platform_type` + `platforms.<TYPE>` + `sensor:` + dalis `fleet:`
- **`platform_type`**: `FIXED_WING | MULTIROTOR | VTOL` — loaderis išskleidžia vieną lentelę į aktyvų `PlatformConfig`.
- **`platforms.<TYPE>`**: `v_cruise, v_coverage, v_climb, v_descent, r_min_m, omega_max, climb_angle_deg, ground_roll_energy_j, mass_kg`, ir `power_w{IDLE,TAKEOFF,CLIMB,CRUISE,COVERAGE,TURN,DESCENT,LAND,HOVER}` (W).
- **`fleet.battery_capacity_wh`** (100 Wh; DJI 99.5), **`fleet.drone_dims_m`**.
- **`sensor`**: `swath_width_m` (100), `overlap_frac` (0.5), `sensor_power_w` (0 = kamera išjungta; įsijungia tik COVERAGE segmentuose).

### c) Spiečius — `fleet:` + `aero:` + `launch:` + `swap:`
| Raktas | Default | Reikšmė |
|---|---|---|
| `fleet.n_drones` | 5 | Dronų kiekis (1–100) |
| `fleet.total_reserve_batteries` | 50 | Bendras swap-paketų fondas; išsekus → MISSION_FAILED (omit = ∞) |
| `aero.*` | — | formation_drag_reduction (0.1514), formation_spacing_m, downwash_*, rth_rendezvous_window_s |
| `launch.candidate_sites` + `w_distance/w_energy/w_swaps` | 8 / 0.3/0.5/0.2 | Starto vietos parinkimo svoriai |
| `swap.service_time_s` / `n_bays` | 90 / 2 | Swap trukmė (TIME, 0 energijos) + lygiagretūs bays |

### d) Kliūtys — `env` (statinės) + `dynamic_obstacles:`
| Raktas | Default | Reikšmė |
|---|---|---|
| `env.obstacle_density_per_km2` | 8.0 | Poisson(density·area) |
| `env.obstacle_size_range_m` | [20,80] | `[S,S]` = fiksuotas dydis |
| `env.obstacle_shapes` | [circle,rectangle,polygon] | + `square` (DJI); nežinoma forma → ConfigError |
| `env.n_obstacle_classes` | 3 | Semantinės klasės (GVG) |
| `env.obstacle_floor_m` / `obstacle_ceil_range_m` | 0.0 / null | 2.5D prizmė; null = neribotos lubos (2D-identiška) |
| `dynamic_obstacles.*` | enabled:false | count, speed_m_s, size_m, passive/active_sense_range_m, active_scan_power_w, dynamic_hold_s |

### e) Kiti — pakartojimai, gedimai, RTH, sim, MC
| Raktas | Default | Reikšmė |
|---|---|---|
| **`failure.hazard_rate_per_hour`** | 0.3 | **λ (gedimų tikimybė)** — vienintelis; 0.0 = išjungta |
| **`mc.n_max` / `n_min` / `ci_tolerance`** | 1000 / 30 / 0.01 | **Pakartojimai** (konvergencija pagal π_time CI) |
| `sim.master_seed` / `dt_s` / `max_timesteps` | 42 / 0.5 / 50000 | Determinizmo kertinis akmuo |
| `battery_zones.high/nominal/critical` | 0.75/0.40/0.20 | Zonos; **RTH statinis slenkstis = nominal 0.40** |
| `rth.check_interval_s` / `reserve_frac` | 5 / 0.05 | + `rth.energy_map.{enabled,decide,route,zone_demotion,cell_m,yellow_penalty,red_threshold}` (visi default-OFF) |
| `safety.*` | — | min_separation_m, obstacle_buffer_m, predict_horizon_s, `obstacle_recovery`, `stall_detector`, `stall_skip` |
| `telemetry` / `viz` / `mission` / `tier_thresholds` | OFF / OFF / coverage / [15,50] | Stebėsena / vizualai / misijos tipas / tier ribos |

---

## 3. Universalūs CLI flag'ai (kartojasi daugelyje skriptų)

| Property | Tipas | Ką reiškia |
|---|---|---|
| `--config` | failas | Kurį YAML įkelti. Nėra merge — vienas failas + dataclass default'ai |
| `--out` / `--base` | katalogas | Bazinis `runs/` katalogas (rezultatai eina į paakatalogį) |
| `--run-name` | string | Fiksuoja run-folder vardą; be jo — unikalus `<vardas>_<timestamp>_<guid>` (perrašymo nėra) |
| `--jobs` | int \| `auto` | Lygiagretūs procesai; `auto` = fiziniai branduoliai − 1. Rezultatas byte-identiškas serial'ui |
| `--resume` | failas/katalogas | Tęsia nutrūkusį run'ą: baigti taškai patikrinami pagal identity, praleidžiami, sujungiami |
| `--profile` | bool | cProfile'ina vieną reprezentatyvų gabalą ir išeina (nepaleidžia tinklelio) |

**Svarbu:** nėra universalaus `--set key=value`. Dauguma fizikos/parametrų (λ gedimai, baterija,
platforma, kliūčių dydis) keičiami **TIK per YAML**; CLI flag'ai eksponuoja tik studijų ašis
(`--n`, `--areas`, `--densities`, `--reps`, `--seed`, `--arms`, `--spares`).

---

## 4. Skriptai

### I tier — viena misija / atkūrimas (demo)

#### `run_single_mission` — gynybos demo: viena misija, pilnas vizualų dump'as
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_single_mission --config config/djimatrice4e.yaml --algo tgc_basic --planner dubins --seed 123 --name defense-demo --base runs
```
**Ką atlieka:** Paleidžia VIENĄ misiją su pasirinktu dekompozicijos algoritmu ir kelio planuotoju,
priverstinai įjungia telemetriją (GPX). Sukuria struktūruotą `runs/run-<ts>/simulation-<name>/`
aplanką su `plan.json`, `results.json` ir visais paveikslais: aplinka, partition, keliai, replay
GIF, state Gantt, baterijos, π-bars, SMDP konvergencija, `tracks.gpx`. Jei įjungtas
`rth.energy_map.decide`, papildomai įrašo per-sortie „arming" slenksčius.

**Sėkmės atveju** stdout baigiasi eilute:
`[single] outcome=MISSION_SUCCESS coverage=1.000 energy=<J> duration=<s> workload_std=<m> | efficiency 0.xxx`
+ konvergencijos lentelė + `run -> runs/.../`. **Interpretacija:** `outcome` turi būti
`MISSION_SUCCESS`, `coverage≈1.0`; `workload_std` = dronų krūvio disbalansas (mažiau geriau);
`efficiency` = SMDP throughput (ne energijos). Nedidelis `coverage<1.0` su `MISSION_PARTIAL` =
kliūčių „boxing".

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--algo` | enum{classic_voronoi, kmeans, tgc_basic, weighted_voronoi} | Zonų dekompozicija |
| `--planner` | enum{dubins, grid} | Kelio planuotojas (dubins = ne-holonominis) |
| `--seed` | int | Perrašo `sim.master_seed` |
| `--name` | string | Simuliacijos poaplankio vardas (default = algo) |
| + universalūs | | `--config --base --run-name` |

#### `run_replay` — atkuria vieną konkrečią replikaciją kaip animaciją
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_replay --config config/scenarios/smoke.yaml --replication 60 --seed 42 --algo weighted_voronoi --planner dubins --fps 12 --max-frames 200 --out runs/replay60
```
**Ką atlieka:** Kadangi variklis determinuotas, bet kurią MC serijos replikaciją galima tiksliai
atkurti nurodžius tą patį `--replication` indeksą (nereikia saugoti trajektorijų). Įrašo
`paths.png` (statiniai keliai, spalvinti pagal būseną) ir `replay.gif` (animacija).

**Sėkmės atveju:** `[replay] replication=60 seed=42 ...` + `aborted=False coverage=0.xxx duration=<s>`
+ keliai į abu failus. **Interpretacija:** naudok „simulaciją #N" iš batch'o vizualiai
išanalizuoti — pvz. kodėl konkreti replikacija žlugo.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--replication` | int | Kurią replikaciją atkurti (default 0) |
| `--seed` | int | Master seed (default — iš config) |
| `--algo` / `--planner` | enum | Kaip run_single_mission |
| `--fps` | int | GIF kadrų/s (default 12) |
| `--max-frames` | int | GIF kadrų riba (default 200) |
| `--out` | katalogas | Išvesties aplankas (default `runs/replay`) |

---

### II tier — Monte-Carlo studijos (pagrindiniai tezės rezultatai)

#### `run_decomposition_comparison` — 4 algoritmų palyginimas (headline)
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_decomposition_comparison --config config/default.yaml --base runs --run-name decomp-headline
```
**Ką atlieka:** Vienas run'as, viena simuliacija KIEKVIENAM iš 4 peer algoritmų
(classic_voronoi, kmeans, tgc_basic, weighted_voronoi) ant PORUOTŲ sėklų (vienas bendras
`RngFactory`). Kiekvienas — CI-konvergencijos MC. Sukuria `workload_box.png` ir per-algo
`plan.json`+`results.json`.

**Sėkmės atveju** — lentelė stdout: `variant  runs  conv  workload_std  duration  energy  efficiency`.
**Interpretacija:** `conv=Y` = MC konvergavo; lygink `workload_std` (balansas) ir `efficiency`.
SVARBU (homogeniška flotilė, λ=0): `weighted_voronoi ≡ tgc_basic` byte-identiškai — tai laukiamas
null, ne bug'as.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| + universalūs | | `--config --base --run-name` (skripto specifinių nėra) |

#### `run_shape_sweep` — S5 formos sweep (centrinis empirinis tinklelis)
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_shape_sweep --config config/default.yaml --mode clean --budget full --shapes square,l_shape,c_shape,star_5 --n 2,3,4,5,6 --jobs 4 --run-name shape_sweep_clean
```
**Ką atlieka:** Sweepina survey FORMĄ (9 vienodo ploto 1 km² poligonai) × flotilės dydį n ×
dekompozicijos variantą ant poruotų sėklų. Homogeniška flotilė, λ=0, kliūtys pagal `--mode`.
Fiksuotas N per ląstelę (ne CI-stopping — kad sėklų poravimas išliktų tikslus). Išveda
`shape_sweep.csv`, `contrasts.csv` (poruoti skirtumai su CI), `summary.md` (su H1/H2
„honest read-out").

**Sėkmės atveju:** `S5 shape sweep: mode=clean N=20 shapes=9 ...` → `run -> runs/.../`. Grąžina
exit 0. **Interpretacija:** headline kontrastai — TGC vs classic_voronoi ir vs kmeans per
(forma, n); `contrasts.csv` CI, neapimantis 0 = reikšmingas skirtumas. Jei `PROBLEM cells`
(exit 1) — žr. `summary.md`.

Kanoninės 9 formos: `square, rect_2_1, rect_4_1, rect_8_1, disk, l_shape, star_5, pinwheel, c_shape`.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--shapes-dir` | katalogas | Kur ieškoti/rašyti formų geojson |
| `--mode` | enum{clean, shipped} | clean = 0 kliūčių (grynas formos efektas); shipped = config tankis |
| `--budget` | enum{quick, full} | Fiksuotas N: quick→5/10, full→20/100 |
| `--n-runs` | int | Perrašo budget'o N per ląstelę |
| `--shapes` | string sąrašas | Formos (default = kanoninės 9) |
| `--n` | int sąrašas | Flotilės dydžiai (default 2..6 clean) |
| + universalūs | | `--config --base --run-name --jobs --profile` |

#### `run_scale_tiers` — smulkus flotilės dydžio sweep + break-even
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_scale_tiers --config config/default.yaml --n-range 2 100 2 --mode clean --budget full --jobs 4 --out runs
```
**Ką atlieka:** Sweepina weighted-TGC prieš kmeans bazinę liniją per flotilės dydžius, kiekvienas
variantas MC su CI-adaptyviu stabdymu (kiekvienas n naudoja tik tiek replikacijų, kiek reikia).
Suranda EMPIRINĮ break-even n* — kur weighted aplenkia kmeans kiekvienai „mažiau-geriau" metrikai.
Rašo `scale_sweep.csv` + plot.

**Sėkmės atveju** — lentelė `n algo runs conv energy_J dur_s wl_std_m eff` + „empirical break-even"
sekcija su `n* = X.X` (arba `no crossing`). **Interpretacija:** `n*` = flotilės dydis, nuo kurio
weighted dekompozicija apsimoka; `conv=Y` visose eilutėse = patikima. `no crossing` = weighted
neaplenkia toje ribose.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--n` | int sąrašas (nargs+) | Konkretūs dydžiai, pvz. `--n 4 8 16 24` |
| `--n-range` | 3×int (START STOP STEP) | Tinklelis; perrašo `--n` |
| `--mode` | enum{clean, shipped} | Kliūtys off/on |
| `--budget` | enum{quick, full} | MC replikacijų lubos |
| + universalūs | | `--config --out --run-name --jobs --profile` |

#### `run_area_obstacle_sweep` — plotas × kliūčių tankis × n (scale eksperimentas)
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_area_obstacle_sweep --config config/djimatrice4e.yaml --areas 1,2,4,8,16 --densities 0,8 --obstacle-size-m 52.8 --n-range 2 8 2 --reps 20 --shapes l_shape --variants tgc_basic,kmeans --jobs 4 --out runs
```
**Ką atlieka:** Sweepina survey PLOTĄ (K1 — formos regeneruojamos kiekvienam plotui, proporcijos
fiksuotos), statinį kliūčių TANKĮ (K4 — count = Poisson(density·area_km2)), fiksuotą kliūties DYDĮ,
ir flotilę (K2). Kiekviena ląstelė = `(shape, area, density, n)`, fiksuotas N poruotų replikacijų.
Default config = DJI M4E. Išveda `area_obstacle_sweep.csv`, `contrasts.csv`, `summary.md`.

**Sėkmės atveju:** `area x obstacle sweep: shapes=[...] areas=[...] ...` → `run -> runs/.../`.
**Interpretacija:** kaip dekompozicijos pranašumas skaluojasi su plotu ir tankiu; `--densities 0`
izoliuoja grynąjį formos/masto efektą. `--obstacle-size-m S` → `obstacle_size_range_m=[S,S]`
(fiksuotas dydis, kintantis kiekis).

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--areas` | float sąrašas | Ploto ašis km² (K1), pvz. 1,2,4,8,16 |
| `--densities` | float sąrašas | Kliūčių tankis /km²; 0 = švaru (K4) |
| `--obstacle-size-m` | float | Fiksuotas dydis S → `[S,S]` |
| `--n` / `--n-range` | int sąrašas / 3×int | Flotilės dydžiai (K2) |
| `--reps` | int | Fiksuotas N per ląstelę (default 20; adaptive OFF) |
| `--shapes` | string sąrašas | Formos (default `l_shape`) |
| `--variants` | string sąrašas | Iš 4 peer + 3 naive-launch variantų |
| `--disk-sides` | int | Disko poligono kraštinės (default 128) |
| `--shapes-dir` | katalogas | (nenaudojamas generavimui; suderinamumui) |
| + universalūs | | `--config --out --run-name --jobs` |

#### `run_rth_ab` — Stage-5 RTH A/B: 4 „rankos" ant poruotų sėklų (C1 rezultatas)
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_rth_ab --config config/study01_demand.yaml --reps 100 --arms 1 2 3 4 --jobs 4 --out runs
```
**Ką atlieka:** Testuoja tezės teiginį: dinaminis, atstumo-suvokiantis energijos-žemėlapio RTH
pranoksta statinį baterijos-frakcijos slenkstį. 4 rankos, kiekviena prideda po vieną dalyką:
(1) static-40% bazinė, (2) route-only, (3) decide+route, (4) full-map (zone_demotion, TERMINAL
grindys 0.10). Skiriasi TIK `energy_map` flag'ais + arm-4 grindimis — `default.yaml`
neredaguojamas. Fiksuoja `reason_out` atribuciją.

**Sėkmės atveju** — per-arm eilutės `arm N rep k/reps: OUTCOME D=<n> rth=<c> crit=<c> term=<c>` +
galutinė lentelė. **Interpretacija:** HEADLINE = perėjimo-priežasties inversija: arm 1 dominuoja
`critical_battery`, arm 4 — `rth_energy`. Ta inversija IR YRA rezultatas. `arm3−arm1` izoliuoja
routing'ą; `arm4−arm3` izoliuoja decide+demotion.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--reps` | int | Poruotos replikacijos per ranką (default 100) |
| `--arms` | int sąrašas {1,2,3,4} | Kurias rankas leisti |
| + universalūs | | `--config` (default `study01_demand.yaml`) `--out --resume --jobs` |

#### `run_spare_sizing` — kiek atsarginių swap-baterijų sandėliuoti
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_spare_sizing --config config/default.yaml --demand-mode --reps 500 --margin 1 --jobs 4 --out runs/spares
```
**Ką atlieka:** Sweepina `fleet.total_reserve_batteries` (bendras baigtinis swap-paketų fondas).
Misija = SĖKMĖ, kai padengimas baigiamas prieš išsenkant fondui. Randa sėkmės-tikimybės „kelį"
(knee) ties 99% ir 95% (Wilson apatinė riba) ir tikrina analitinį prior
`spares ≈ E_cover/B_usable − n + margin`. **Demand mode** (`--demand-mode`): vietoj B-tinklelio —
vienas neribotas batch'as, matuojama kiekvienos replikacijos paklausa D, o visa kreivė
rekonstruojama post-hoc (O(reps) vietoj O(tinklelis×reps)).

**Sėkmės atveju** — knee lentelė + `[structured output: runs/...]` + `spare_sizing_knee.png`.
**Interpretacija:** knee = mažiausias paketų skaičius, kurio Wilson-apatinė riba viršija tikslą.
Įspėjimas jei `--reps` < 381 (99% negali būti *sertifikuotas*, tik point-estimate).

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--reps` | int | Poruotos replikacijos per tašką (default 200) |
| `--margin` | int | Papildomas paketų rezervas prior formulėje |
| `--spares` | int sąrašas (nargs+) | Eksplicitiniai kiekiai (išskiriamasis su --spare-range) |
| `--spare-range` | 3×int | Kiekių tinklelis (imtinai STOP) |
| `--span` | int | Sweep bracket'o pusplotis apie prior (default 8) |
| `--demand-mode` | bool | Vienas neribotas batch'as vietoj tinklelio |
| + universalūs | | `--config --out --resume --jobs` |

---

### III tier — analitiniai sprendimų įrankiai (BE simuliacijos)

#### `run_fleet_sizing_analyzer` — flotilės dydžio Pareto (mažėjančios grąžos)
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_fleet_sizing_analyzer --config config/djimatrice4e.yaml --n-min 1 --n-max 20 --knee-frac 0.05 --plot runs/pareto.png
```
**Ką atlieka:** Pastato planavimo sluoksnį (EnvironmentMap + energijos-suvokiantis
LaunchSiteOptimizer), kad gautų TIKRĄ bazės pozą ir navigacinį plotą, tada per gryną
`fleet_sizing` branduolį atspausdina „mažėjančios grąžos" Pareto lentelę. NELEIDŽIA simuliacijos —
tai priešpaskaičiavimas prieš sunkų run'ą.

**Sėkmės atveju** — Markdown antraštė su prielaidomis + lentelė
`N | Est. Mission Time | Est. Total Swaps | Time Saved vs N-1` su `← knee` žyma.
**Interpretacija:** knee = N, nuo kurio kiekvienas papildomas dronas taupo < `knee-frac`
vienos-drono trukmės → renkis tą N ir įrašyk į YAML rankomis. `MISSION IMPOSSIBLE` (exit 2) =
joks starto taškas neįmanomas.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--n-min` / `--n-max` | int | Vertinamų flotilės dydžių ribos (default 1..20) |
| `--knee-frac` | float | Mažėjančios grąžos slenkstis (dalis N=1 trukmės, default 0.05) |
| `--plot` | failas | Nebūtinas Pareto PNG kelias |
| + universalūs | | `--config` (default `djimatrice4e.yaml`) |

#### `run_regime_calculator` — A2 režimas: E_cover vs n·B_usable
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_regime_calculator --config config/default.yaml --geojson data/areas/shapes/l_shape.geojson --n-drones 4 --usable-floor terminal --sensor-power-w 15 --verify --verify-n 3
```
**Ką atlieka:** Atsako į go/no-go klausimą, kurį formos studija privalo išspręsti PIRMA: ar bazė
**battery-limited** (reikia swap'ų → forma svarbi) ar **fuel-surplus** (vienas įkrovimas padengia
zoną → forma nustoja moduliuoti). Naudoja DU testus: pooled ratio (apatinė riba) + užimčiausio
drono zonos energija prieš vieną bateriją. `E_cover` — ne fudge formulė, o ta pati fizika kaip
simuliatoriuje. `--verify` palygina su realia misija.

**Sėkmės atveju** — Markdown su E_cover išskaidymu (strips + connectors + camera + transit +
vertical), `B_usable` ir režimo klasifikacija. **Interpretacija:** jei pooled ratio > 1 → tikrai
battery-limited (kuris nors dronas privalo swap'inti); per-drono max-zone ratio > 1 → užimčiausias
dronas battery-limited (PIRMINIS rodiklis). Jei fuel-surplus — formos sweep būtų plokščias.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--geojson` | failas | Survey poligonas (default — iš config) |
| `--n-drones` | int | Flotilės dydis (default — iš config) |
| `--usable-floor` | enum{terminal, return, rth} | Kuris rezervo lygis → B_usable (0.20/0.40/reserve_frac) |
| `--sensor-power-w` | float | Kameros galia (perrašo `sensor.sensor_power_w`) |
| `--verify` | bool | Palygink analitinę su realios misijos energija |
| `--verify-n` | int | Verifikacijos replikacijų (default 1) |
| + universalūs | | `--config` |

#### `run_shape_regime_table` — formos-režimo lentelė (partition picture)
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_shape_regime_table --config config/default.yaml --n-min 1 --n-max 6 --f-min 0.40 --f-max 1.00 --perms 8 --usable-floor terminal --csv runs/shape_regime.csv
```
**Ką atlieka:** Gryna analizė virš esamų A2/A3 įrankių (jokio MC). Kiekvienai `(forma, n)` ląstelei —
du režimo rodikliai (pooled + per-drono max-zone) prieš tikrą weighted-Voronoi partition, ir
formos sukeltas zonų disbalansas. Atsako į 4 klausimus prieš MC: n* lentelė, weighted vs
unweighted redistribucijos momentu, H5 signalas (solidity vs isoperimetric), sweep-grid dizainas.

**Sėkmės atveju** — lentelės stdout + nebūtinas CSV. **Interpretacija:** disbalansas seka
ISOPERIMETRIC ratio (pailgumą), NE solidity — plonas iškilus rect_8_1 partition'inasi blogiau nei
įgaubtas star_5. Tikrasis rezultatas: weighted dekompozicija labiau sumažina užimčiausio drono
krūvį įgaubtoms formoms.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--n-min` / `--n-max` | int | Flotilės diapazonas (default 1..6) |
| `--f-min` / `--f-max` | float | Baterijos-frakcijos diapazonas redistribucijos profiliui |
| `--perms` | int | Permutacijų skaičius (default 8) |
| `--usable-floor` | enum{terminal, return, rth} | Kaip regime_calculator |
| `--sensor-power-w` | float | Kameros galia |
| `--csv` | failas | Rašo pilną per-ląstelę tinklelį |
| + universalūs | | `--config --shapes-dir` |

#### `run_launch_site_study` — starto vietos optimizacija (§2.4)
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_launch_site_study --config config/default.yaml --out runs/launch
```
**Ką atlieka:** Pastato variklio build fazę (kuri apskaičiuoja starto vietą ir kandidatų balus) ir
atspausdina reitinguotą kandidatų lentelę. Naudoja `launch.w_distance/w_energy/w_swaps` svorius.

**Sėkmės atveju:** `chosen launch site: (x, y)` + top-10 lentelė
`site mean_dist energy exp_swaps J`. **Interpretacija:** mažesnis `J` = geresnė vieta;
`exp_swaps` = tikėtini swap'ai iš tos vietos. Pasirinktoji = mažiausio J.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--out` | katalogas | (default `runs/launch`) |
| + universalūs | | `--config` |

#### `run_kinematics_comparison` — Dubins vs grid (tik FW/VTOL, §1.2)
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_kinematics_comparison --config config/default.yaml --out runs/kinematics
```
**Ką atlieka:** Palygina Dubins vs diskretizuotą grid planuotoją. **Tik ne-holonominėms
platformoms** — MULTIROTOR grąžina exit 1 su žinute (holonomiškai beprasmiška; nustatyk
`platform_type: FIXED_WING` arba `VTOL`).

**Sėkmės atveju:** lentelė `planner plan_time duration energy`. **Interpretacija:** lygink
planavimo laiką vs skrydžio kokybę tarp dviejų planuotojų.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--out` | katalogas | (default `runs/kinematics`) |
| + universalūs | | `--config` (privalo būti FW arba VTOL) |

---

### IV tier — utilitos (generavimas, žemėlapiai, diagnostika)

#### `generate_shapes` — A3 vienodo ploto formų generatorius
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.generate_shapes --config config/default.yaml --target-area-m2 1000000 --out-dir data/areas/shapes --disk-sides 128
```
**Ką atlieka:** Sugeneruoja formų šeimą (kanoninės 9), visas su TUO PAČIU plotu (default 1 km²) bet
skirtinga forma, kaip GeoJSON simuliatoriaus loaderio formatu. Vienodas plotas izoliuoja formos
efektą (strip darbas ~ plotas/swath yra formos-invariantiškas).

**Sėkmės atveju** — deskriptorių lentelė: plotas, perimetras P, isoperimetric ratio, hull plotas,
solidity, bbox, strip skaičius + įrašyti geojson. **Interpretacija:** `solidity < 1` = įgaubta
(L/star/pinwheel — H5 kintamasis); isoperimetric > 1 = pailga.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--target-area-m2` | float | Tikslinis plotas (default 1_000_000) |
| `--out-dir` | katalogas | Kur rašyti geojson (default `data/areas/shapes`) |
| `--disk-sides` | int | Disko poligono kraštinės (default 128) |
| + universalūs | | `--config` (skaito swath/overlap strip skaičiavimui) |

#### `plot_launch_suitability` — B6.3 starto-tinkamumo šilumos žemėlapis
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.plot_launch_suitability --config config/djimatrice4e.yaml --out runs/launch_suitability.png
```
**Ką atlieka:** Piešia per staging žiedą (aplink survey plotą) kiekvieno starto taško kokybę pagal
TIKSLŲ flotilės nuovargį (swap skaičių), naudodamas tą pačią matematiką kaip optimizatorius.
Ląstelės už survey ribų (GCS negali būti poligono viduje), spalvintos pagal swap zoną.

**Sėkmės atveju** — PNG su žiedu apie nespalvotą tikslinį poligoną. **Interpretacija:** žalia = 0
swap'ų (idealu), geltona/oranžinė/raudona = augantis nuovargis. Rodo, kur optimizatorius rinktųsi.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--out` | failas | PNG kelias (default `launch_suitability_map.png`) |
| + universalūs | | `--config` |

#### `run_llm_diagnosis` — Phase-4 LLM-as-judge diagnozė virš telemetrijos
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.run_llm_diagnosis --log runs/demo/events.jsonl --out runs/demo/diagnosis.md --model claude-sonnet-4-6 --max-tokens 2000 --max-events 400 --dry-run
```
**Ką atlieka:** Paleidžia LLM diagnozę virš telemetrijos JSONL log'o. Visada paremta determinuotų
faktų bloku + grounding auditu, todėl LLM haliucinacija (neteisingas outcome, neegzistuojantis
dronas) pažymima, ne aklai patikima. `--dry-run` parodo faktus + tikslų prompt'ą BE API kvietimo
(nemokamai).

**Sėkmės atveju:** `[diagnosis] outcome=... root_cause=... confidence=0.xx grounded=True checks=P/T`.
**Interpretacija:** `grounded=True` + visi `checks` praėję = patikima; `confidence` = LLM tikrumas.
Realiai diagnozei reikia `ANTHROPIC_API_KEY` (arba `--api-key`).

| Property | Tipas | Paaiškinimas |
|---|---|---|
| `--log` | failas (privalomas) | Telemetrijos events JSONL |
| `--out` | failas | Markdown ataskaitos kelias |
| `--model` | string | Anthropic modelio ID |
| `--max-tokens` | int | Atsakymo riba (default 2000) |
| `--max-events` | int | Log įvykių riba modeliui (default 400) |
| `--api-key` | string | Perrašo `ANTHROPIC_API_KEY` |
| `--dry-run` | bool | Parodo faktus+prompt, NEkviečia API |

#### `bench_separation` — B3 separacijos mikro-benchmark'as
**Pavyzdys:**
```bash
python -m uav_swarm_sim.experiments.bench_separation
```
**Ką atlieka:** Matuoja seną O(n²) porinį separacijos skenavimą prieš naują
`SafetyMonitor._separation_yielders` (KDTree) didėjančiuose flotilės dydžiuose ir tvirtina, kad
abu duoda identišką rezultatą. **Argumentų neturi.**

**Sėkmės atveju** — lentelė `n | O(n^2) ms/tick | KDTree ms/tick | speedup | identical`.
**Interpretacija:** `identical=True` visose eilutėse = refaktoringas korektiškas; `speedup` auga su n.

| Property | Tipas | Paaiškinimas |
|---|---|---|
| (nėra) | — | Skriptas be CLI argumentų |

---

## 5. Kaip renkiesi

- **Vieną misiją pamatyti / demo** → `run_single_mission`; **atkurti konkrečią** → `run_replay`
- **Algoritmų / formos / masto / RTH / atsargų MC** → `run_decomposition_comparison`,
  `run_shape_sweep`, `run_scale_tiers`, `run_area_obstacle_sweep`, `run_rth_ab`, `run_spare_sizing`
- **Priešpaskaičiavimai be simuliacijos** → `run_fleet_sizing_analyzer`, `run_regime_calculator`,
  `run_shape_regime_table`, `run_launch_site_study`, `run_kinematics_comparison`
- **Įrankiai** → `generate_shapes`, `plot_launch_suitability`, `run_llm_diagnosis`, `bench_separation`
