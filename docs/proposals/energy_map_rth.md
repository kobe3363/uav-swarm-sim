# Energy-map RTH & routing — dizaino pasiūlymas (EM-01)

> **Statusas:** PASIŪLYMAS — kodas NEkeičiamas šiuo dokumentu. Įgyvendinimas
> vyksta atskiromis, flag'uotomis, byte-identity-gated stadijomis (§9) po
> autoriaus GO. Dokumentas savarankiškas: kiekvienas teiginys apie dabartinį
> kodą turi šaltinį (`failas:eilutė`); įverčiai be tiesioginio kodo šaltinio
> pažymėti **ESTIMATE** su formule ir prielaidomis; autoriaus diagnozės (ne repo
> artefaktai) pažymėtos **AUTORIAUS DIAGNOZĖ**.
>
> Data: 2026-07-17 · bazinis commit: `30f4209` (main)

---

## 0. Santrauka

Šiandien tezės skelbiamas **dinaminis RTH** (`rth_calculator.decide`) numatytoje
1 km² srityje praktiškai neveikia: jį pre-emptina statinis **40%** baterijos
slenkstis (CRITICAL guard'as ties `nominal=0.40`, §1). Grįžimo (S3_RTH) ir resume-transit keliai planuojami kaip tiesios
stygos su reaktyviu 15 m sidestep, todėl kliūtys sukelia obstacle-boxing livelock
ir riboja success rate (§1). O `path_clear` tikrinimas kas 5 s yra #1 CPU hotspot.

Sprendimas (autoriaus priimtas): **per-replikacijos energijos cost-to-go grid'as
(„energy map")** — vienas Dijkstra nuo bazės virš occupancy costmap'o, saugantis
per-cell `E_home` (J, grįžimo kaina namo) + parent pointer. Jis vienu metu:
(1) paverčia RTH sprendimą poziciškai tiksliu ir obstacle-aware; (2) suteikia
obstacle-avoiding grįžimo ir resume kelius vietoje tiesių stygų; (3) pašalina
per-tick `path_clear` hotspot'ą (vienas build vietoje O(N) tikrinimų). Visa už
flag'ų, default OFF, byte-identity garantija.

Šis dokumentas detalizuoja autoriaus priimtus sprendimus (§§1–4), specifikuoja
build'ą ir penkis integration seam'us (§§5–8), pateikia flag'avimo/rollout planą
(§9), A/B eskizą (§10) ir sąžiningas ribas (§11).

---

## 1. Motyvacija

### 1.1 Dinaminis RTH nepastebimas: efektyviai valdo statinis 40% slenkstis

Coverage'o pertraukimo guard'ai vertinami tvarka (pirmas atitikmuo laimi,
`state_machine.py:109-116`):

```
obstacle_threat (:109-110)  ->  rth_energy (:111-112)  ->  critical_battery <0.40 (:113-114)  ->  terminal_battery <0.20 (:115-116)
```

**Svarbi pataisa ankstesniam framing'ui:** pre-emption įvyksta vos `f` nukrenta
žemiau **0.40**, NE ties 0.20. `critical_battery` guard'as suveikia kai
`battery_zone is CRITICAL`, o CRITICAL yra juosta `[critical, nominal) =
[0.20, 0.40)` (`battery.py:44-50`, `config/default.yaml:215-216`,
`config.py:155-159`) — tad guard'as kerta grįžimą juostos VIRŠUJE, ties
`nominal=0.40`, ne apačioje. `nominal=0.40` NĖRA vien reporting bin — tai
efektyvusis statinio grįžimo slenkstis (`run_regime_calculator.py:48-52`); 0.20
yra tik TERMINAL pradžia. Guard'ų tvarkoje `rth_energy` (dinaminis) testuojamas
**pirmas** (`state_machine.py:111-112`), tad problema yra ne tvarka, o tai, kada
`should_return` tampa `True`.

Autoriaus motyvacija (parafrazuota, patikslinta ties 0.40 — NE verbatim):

> Dinaminis RTH (`should_return`: `level < E_home + e_next + reserve_frac·cap`)
> guard'ų tvarkoje testuojamas pirmas (`state_machine.py:111-112`), bet 1 km²
> plotuose grįžimo kaina maža, todėl jo slenkstis pasiekiamas tik ~20% baterijos
> — GILIAI žemiau statinio 0.40 net'o. Todėl statinis `critical_battery` tinklas
> (0.40) suveikia gerokai anksčiau ir grąžina droną, kol dinaminis trigeris (~20%)
> dar nepasiektas: dinamiškumas lieka nepastebimas, efektyviai misiją valdo
> statinis 40% slenkstis. Tai tiksliai motyvuoja scale ašį — didėjant plotui
> grįžimo kaina auga, `should_return` slenkstis pakyla virš statinio 0.40 net'o,
> ir dinaminis RTH pradeda realiai dominuoti. Energijos žemėlapio indėlis:
> padaryti tą dominavimą poziciškai tikslų ir obstacle-aware visuose masteliuose,
> o ne priklausomą nuo to, ar grįžimo kaina atsitiktinai viršija statinį slenkstį.

Skaitmeninis pagrindas (`should_return`, `rth_calculator.py:72-82`):
`reserve = reserve_frac·cap = 0.05·360 000 = 18 000 J` (`reserve_frac=0.05`
`config/default.yaml` rth; reserve skaičiuojamas `rth_calculator.py:80`). 1 km²
srityje grįžimo kaina nuo giliausio taško
(~1 km × 18.333 J/m ≈ 18 kJ + landing) plius vieno coverage leg'o bundle yra maža
capaciteto (360 kJ) atžvilgiu, todėl `level < e_next + return + reserve` sąlyga
išsipildo tik ties ~20% — t.y. giliai žemiau statinio **0.40** net'o, tad statinis
grįžimo guard'as (0.40) suveikia gerokai anksčiau ir dinaminį trigerį užstoja.

### 1.2 Tiesios stygos + reaktyvus sidestep → obstacle-boxing

- **S3_RTH grįžimas** planuojamas kaip tiesi styga:
  `ret = motion.plan(self.pose, self.base, CRUISE)` (`agent.py:321-323`).
- **Resume-transit** po swap: `motion.plan(base, entry, CRUISE)` (arba FIX-B1
  `route_transit`, jei įjungta; `agent.py:390-398`).
- **Kliūties vengimas** yra reaktyvus 15 m šoninis šuolis
  (`_avoidance_plan`, `agent.py:400-406`: `15*lx`, `15*ly`, `+10*cos/sin`).

Kai styga kerta kliūtį, runtime kelia S_OBS ir šokinėja į šoną; po
`_OBS_REENTRY_BUDGET = 6` (`agent.py:50`) nesėkmingų re-entry drone'as pripažįstamas
„boxed in" ir siunčiamas namo (`agent.py:342-344`). Tai yra obstacle-boxing
livelock'o mechanizmas.

**AUTORIAUS DIAGNOZĖ (STUDY-01):** ~92.8% success lubos (36/500 INCOMPLETE),
diagnozuotos kaip obstacle-boxed coverage/transit. Pastaba: `docs/reports/` šiuo
metu tuščias (readout NEmerge'intas), tad šie skaičiai cituojami kaip autoriaus
diagnozė, ne repo artefaktas (analogiškai `scale_sweep_v2.md` §0 traktuoja 53 h
incidentą). `config/study01_demand.yaml` egzistuoja.

### 1.3 `path_clear` kas 5 s — #1 CPU hotspot

RTH check kartojasi periodiškai: `(t - _last_rth_t) >= check_interval_s` (5 s),
tik S2_MISSION/S_FERRY būsenose (`agent.py:225-227`). Kiekvienas check kviečia
`return_energy` → `env.path_clear(route)` (`rth_calculator.py:66-69`). Pačios
`first_obstruction`/`path_clear` docstring įvardija tai kaip **„the RTH-lookahead
hot path (~63% of mission runtime)"** (`environment_map.py:157-158, 186-193`).

**AUTORIAUS DIAGNOZĖ:** FIX-B1 visibility routing kainuoja 347–791 s/rep build'e
(memory; ne repo artefaktas). Energy map pakeičia O(N) per-tick `path_clear`
vieninteliu per-rep Dijkstra (§6).

---

## 2. Sprendimas: energijos cost-to-go grid'as

Vienas Dijkstra nuo bazės (`launch_pose`) per grid'ą, kurio kaimynystės briaunų
svoriai = reali cruise energijos kaina × occupancy penalty (§4). Rezultatas —
per-cell `E_home` (mažiausia energija grįžti namo iš to langelio) + parent
pointer (kitas žingsnis namo linkme). Dijkstra, minimizuojantis pilną energiją,
grąžina „žaliausią įmanomą" kelią be papildomos logikos: spalvos (occupancy) YRA
svoriai.

**Literatūros nota (nesudaro naujo indėlio).** Cost-to-go grid'ai ir occupancy
costmap'ai yra klasikinė robotika; tiesioginis analogas — ROS `costmap_2d`
(inflation layer'is → per-cell cost → planner minimizuoja sumą). Šio darbo
indėlis nėra pats grid'as, o **battery-normalizuota, per-replikacijos energijos
map, keičianti statinį RTH slenkstį**, patikrinta paired-seed A/B (§10). Tai
pabrėžtina tezėje, kad recenzentas neįvertintų grid'o kaip pretenduojančio į
naujumą.

---

## 3. Grid rezoliucija (sprendimas 1) — battery-tied

**Taisyklė (fiksuota):** cell edge = atstumas, kainuojantis lygiai 1/1000
baterijos capaciteto REFERENCE CRUISE greičiu. Pagrindimas fiksuotas: map matuoja
GRĮŽIMO kainą, o grįžimai skrenda cruise — ne savavališkas mastelis.

**Aritmetika (AC2):**
- Capacitetas: `battery_capacity_wh=100.0` (`config/default.yaml` fleet) × 3600
  (`WH_TO_J`, `config.py:29, 365`) = **360 000 J**.
- Reference cruise: MULTIROTOR `CRUISE=220.0` W, `v_cruise=12.0` m/s
  (`config/default.yaml` platforms.MULTIROTOR) → **18.333 J/m**
  (`energy_model.distance_energy = P·dist/v`, `energy_model.py:106-118`).
- Cell edge = (360 000 / 1000) J / 18.333 J/m = **360 J / 18.333 J/m = 19.64 m
  ≈ 20 m.**

**STRIKTAI battery-tied, NE obstacle-tied.** Rezoliucija priklauso TIK nuo
baterijos ir cruise kainos. JOKIO ryšio su kliūties dydžiu (pvz. NE
`min(battery, obstacle_radius/2)`): realios kliūtys yra savavališko dydžio, todėl
aparatas negali priklausyti nuo mūsų `obstacle_size_range_m=[20,80]`
(`config/default.yaml` env) konvencijos.

**Grid dimensijos L-shape'ui** (`data/areas/shapes/l_shape.geojson`, nominalus
plotas 1 000 000 m², bbox 1154.7×1154.7 m, fill 0.750; bbox skalė = √A;
cell 20 m; 8-connected → ~4 briaunos/langelį):

| A (km²) | bbox (m) | grid | langeliai | ~briaunos |
|---:|---:|---:|---:|---:|
| 1 | 1155×1155 | 58×58 | 3 364 | 13 456 |
| 2 | 1633×1633 | 82×82 | 6 724 | 26 896 |
| 4 | 2309×2309 | 116×116 | 13 456 | 53 824 |
| 8 | 3266×3266 | 164×164 | 26 896 | 107 584 |
| 16 | 4619×4619 | 231×231 | 53 361 | 213 444 |

(Užduoties „~50×50 → 200×200" atitinka; tikslūs skaičiai 58→231. L-shape notch =
25% bbox langelių krenta už polygono → traktuojami kaip not-in-area / red, §5.)

**Dubins/turn overhead = antros eilės, absorbuojamas reserve.** Holonomic
multirotor (study platforma, `motion_model.py:140-170`) prideda tik in-place-yaw
TURN-power segmentus prie strip galų; grįžimai yra grynas CRUISE, tad map matuoja
tikslią cruise grįžimo kainą. Kvantavimo klaida ≤ 1 langelis = 360 J = 0.1%
capaciteto — įskaičiuota į reserve (§7b, §7). FW/VTOL Dubins arkų overhead būtų
irgi antros eilės ir absorbuojamas to paties reserve.

**Kaimynystė: kvadratinė 8-connected su √2 diagonaliniais svoriais** (pasirinkta).
Diagonalinė briauna kainuoja √2 × cell-cruise-energijos, tad įstrižas kelias nėra
dirbtinai pigus. Heksagonalinis grid'as duotų geresnę izotropiją (mažesnį grid
anisotropy artefaktą kelio kryptims), bet paliekamas kaip **future option — NE
įgyvendinti** šiame darbe.

---

## 4. Occupancy costmap (sprendimas 2) — red-yellow-green = svoriai

Per-cell occupancy fraction → traversal svoris:

| Spalva | Occupancy fraction | Svoris | Prasmė |
|---|---|---|---|
| Green | `f < yellow_thr` | ×1.0 | laisva |
| Yellow | `yellow_thr ≤ f < 0.5` | ×1.5 | sub-cell kliūtis, praeinama su bausme |
| Red | `f ≥ 0.5` | ∞ (blokuota) | netraversuojama |

Dijkstra, minimizuojantis pilną energiją, natūraliai renkasi žaliausią įmanomą
kelią — jokios papildomos logikos: **spalvos YRA svoriai**. Analogas: ROS
`costmap_2d` inflation → per-cell cost.

**Yellow penalty = 1.5 (fizinis argumentas, ne skonis).** Yellow langelis
priverčia sub-cell šoninį apėjimą aplink kampe kertančią kliūtį. Modeliuojame
tikėtiną detour'ą kertant 20 m langelį: kai kampinė kliūtis užima fraction'ą
`φ ∈ (0, 0.5)`, laisvo koridoriaus plotis mažėja nuo pilno langelio link pusės
langelio, ir drone'as turi bowin'ti aplink. Tiesaus perėjimo ilgis `d = 20 m`;
šoninis nuokrypis, reikalingas išlaikyti tarpą, apytiksliai proporcingas `φ·w`,
o papildomas kelio ilgis — atitinkamo dviejų-segmentų apėjimo perteklius. Vidurkis
per `φ ∈ (0, 0.5)` (nuo ~×1.0 vos kertant iki ~×2.0 near-blocked) duoda tikėtiną
multiplikatorių **~×1.5**. Todėl:
- **1.5** = tikėtinos vertės (expected sub-cell sidestep) pasirinkimas — rekomenduoju;
- **2.0** = konservatyvi near-blocked riba (viršutinė intervalo `[1.5, 2.0]` reikšmė).

Reikšmė lieka study-knob'as flag'e (§9), bet default = 1.5 su šiuo argumentu.

---

## 5. Blocked-cell rule (sprendimas 6) — red vs yellow

**Occupancy fraction = `area(cell ∩ buffered_union) / area(cell)`**, kur
`buffered_union` = `env.buffered_obstacles` (clearance-buffered kliūčių sąjunga,
`environment_map.py:72-80`). **Buffered, NE raw** — nes esami maršrutizatoriai
(`route_connector`/`route_transit`) ir `path_clear` operuoja su buffered union
(`visibility_router.py:189`, `environment_map.py:186-193`), o S_OBS trigger'is
naudoja RAW union (`segment_in_obstacle`, `environment_map.py:120-136`).
Naudodami buffered, gauname kelius, gerbiančius regulacinę `clearance_buffer_m=5 m`
maržą (`config/default.yaml` env) ir todėl niekada nekeliančius S_OBS — tas pats
konsistencijos principas kaip esamas router.

Skaičiavimas: tiksli Shapely area-sankirta per langelį (Shapely 2.x batch su
`STRtree`), arba k×k sub-sampling. Rekomenduoju tikslią area-sankirtą (švaresnė,
Shapely 2.x greita). Tai **generalizuoja jau egzistuojantį**
`EnvironmentMap.occupancy_grid(cell_m)` (`environment_map.py:195-205`), kuris
šiandien grąžina cell-center bool vs `free_space` — energy map keičia į fraction.
`GridFrame` (`environment_map.py:25-37`: origin, cell_m, nx, ny, `cell_center`,
`world_to_cell`) naudojamas kaip yra.

**Red slenkstis ≥ 0.5 (pasirinkta).** Konservatyvumo vs corridor-loss analizė
ties 20 m rezoliucija:
- Jei red = „bet koks persidengimas" → per konservatyvu: viena maža kliūtis
  užblokuotų visą 20 m langelį ir uždarytų 20 m pločio koridorių, kurio realiai
  užtenka praskristi (drone bbox ~1.2 m, `config/default.yaml` fleet).
- Jei red = „center inside" → per liberalu: kliūtis, dengianti 49% langelio bet
  ne centrą, liktų žalia, ir Dijkstra vestų kelią beveik per ją.
- **≥50% ploto** yra balansas: langelis, kurio pusė ar daugiau užimta buffered
  kliūties, neturi patikimo pravažiavimo → red; iki 50% → yellow (praeinama, bet
  ×1.5). Ties 20 m grid'u tai išlaiko ~≥10 m laisvo koridoriaus yellow langeliuose
  (daug > drone bbox), o corridor-loss atsiranda tik ten, kur laisvo tarpo < ~10 m
  — kur ir turi būti konservatyvu.

Not-in-area langeliai (už survey polygono / L-shape notch) traktuojami kaip red
(nepasiekiami grįžimui grid'e; realus skrydis už polygono ferry metu lieka §11
riba).

---

## 6. Per-cell saugojimas + build (sprendimai 3, 5)

### 6.1 Saugojimas: TIK `E_home` + parent pointer

Per langelį: `E_home: float` (J; grįžimo namo kaina) + `parent: int` (kito
langelio flat indeksas namo linkme; `-1` bazei). **JOKIŲ** per-cell entry/exit
poses, JOKIŲ precomputed kelių. Kompaktiška: du masyvai `float64[nx*ny]` +
`int32[nx*ny]` (16 km²: ~53k langelių → ~0.6 MB).

Route materializacija vyksta TIK `decide()` trigger'io metu (ne build'e): sekant
parent pointer'ius nuo dabartinio langelio bazės link → waypoint polyline → vienas
smoothing pass per esamą motion model. Multirotor'ui „smoothing" = `HolonomicModel.plan`
(in-place-yaw + straight, `motion_model.py:143-167`) kiekvienai poros grandžiai;
FW/VTOL'ui = `DubinsModel.plan` (`motion_model.py:129-134`). Polyline→Path
grandinės pavyzdys jau egzistuoja: `_chain_cruise_legs` (`visibility_router.py:207-225`)
lygiai taip chain'ina `motion.plan(v_i, v_{i+1}, CRUISE)` į vieną Path su heading
treatment'u — nauja materializacija replikuoja šį seam'ą.

### 6.2 Build: vienas Dijkstra per replikaciją

Kliūtys statinės visą misiją ir generuojamos per replikaciją:
`obs_rng = rng.stream(STREAM_OBSTACLES, replication)` →
`generate_obstacles(area, cfg.env, obs_rng)` + `LayerStack(...)`
(`simulation_engine.py:175, 181-185`). Todėl map statomas **kartą per
replikaciją**, iškart kai bazė žinoma: po `self.launch_pose` nustatymo
(`simulation_engine.py:203`), naudojant layer-0 `self.env` (`:190`).

Briaunų svoriai = reali cruise energijos kaina hop atstumui × occupancy penalty
(§4): green hop = `distance_energy(edge_len, CRUISE, v_cruise)`
(`energy_model.py:106-118`) = 18.333 J/m × edge_len (20 m ortogonaliai, 28.28 m
diagonaliai); yellow hop = × 1.5; red = neįtraukiama. Dijkstra šaltinis = bazės
langelis; `E_home[cell]` = mažiausia suma iki bazės.

**Build-cost ESTIMATE** (formulės + prielaidos):
- Occupancy (Shapely area-sankirta per langelį, batch/STRtree): ~O(cells);
  16 km² ~53k langelių → **~1–5 s** (Shapely 2.x vektorizuota; dominuoja).
- Dijkstra (`scipy.sparse.csgraph.dijkstra`, C): 16 km² V≈53k, E≈213k →
  O(E log V) ≈ 213k × 15.7 ≈ 3.3M op → **~tens ms**. Pure-Python `heapq`
  fallback: **~0.2–1 s**.
- Suma 16 km²: **~1–5 s/rep ESTIMATE**, prieš **AUTORIAUS DIAGNOZĘ** FIX-B1
  visibility routing 347–791 s/rep — t.y. ~2 eilių pigiau, ir vyksta VIENĄ kartą,
  o ne per-tick.

| A (km²) | langeliai | briaunos | Dijkstra (scipy) | build suma ESTIMATE |
|---:|---:|---:|---:|---:|
| 1 | 3 364 | 13 456 | <5 ms | ~0.1–0.3 s |
| 4 | 13 456 | 53 824 | ~10 ms | ~0.3–1 s |
| 16 | 53 361 | 213 444 | ~30 ms | ~1–5 s |

---

## 7. decide() cadence (sprendimas 4) — battery-quantized

Vietoje laiko (5 s) cadence tampa **baterijos-kvantuotas**: žemiau per-sortie
arming slenksčio (§7b) energijos sprendimas vertinamas kas **1% baterijos
kritimo**. 1% capaciteto = 3 600 J = 360 J × 10 ≈ 10 langelių... — tiksliau: 1
langelio hop = 360 J = 0.1% capaciteto, tad 1% = ~10 hop'ų. Užduoties formuluotė
„1% = one cell hop" laikoma dizaino kvantu: sprendimą priimame taip dažnai, kad
kvantavimo klaida ≤ vienas vertinimo intervalas; konservatyviai imame 1%
žingsnį, o jo ≤ 1-langelio poziciją paklaidą įskaičiuojame į reserve (§7b).

Kliūčių grėsmės (S_OBS) lieka **asinchroninės** — cadence valdo TIK energijos
sprendimą, ne threat handling'ą (`signal_threat`, `agent.py:196-207`).

Senasis 5 s laiko kelias (`agent.py:225-227`, `check_interval_s`) **paliekamas už
flago** kaip static-RTH baseline A/B ranka (§10).

---

## 8. Integration seams (7a–7e)

### 7a. RTH decide — pakeičia per-tick plan+path_energy+path_clear
Vietoje `return_energy` (kuris kviečia `motion.plan` + `path_energy` +
`path_clear`, `rth_calculator.py:63-70`) grąžinti, kai:

```
level < E_home[cell(pose)] + e_next_bundle + reserve
```

`e_next_bundle` = `agent.lookahead()` (next COVERAGE leg + following connector +
camera term, `agent.py:411-429`). `cell(pose)` = `GridFrame.world_to_cell`
(`environment_map.py:36-37`). **×1.5 obstacle fudge PAŠALINAMAS**
(`rth_calculator.py:68-69`) — map yra obstacle-aware konstrukciškai (occupancy jau
briaunų svoriuose), tad fudge tampa dvigubu skaičiavimu. Touch point:
`rth_calculator.return_energy`/`should_return` (`rth_calculator.py:63-82`), už map
flago.

### 7b. Arming threshold — perskaičiuojamas kas S0→S1 launch (per sortie)
Kiekvieno sortie likęs planas skiriasi (po swap gilesnis resume, kitas max
grįžimo taškas), tad arm perskaičiuojamas per launch. Formulė:

```
arm = (max E_home over remaining-plan cells + max leg bundle + reserve) / capacity + delta
```

`delta` = 1-langelio (360 J = 0.1% cap) kvantavimo marža (§3, §7). Virš `arm`
skip'inam check'us.

**Bound įrodymas (decisions unaffected).** Jei `level > arm·capacity`, tada pagal
apibrėžimą `level > max_cell E_home + max_bundle + reserve ≥ E_home[current] +
e_next_bundle + reserve` bet kuriam esamam langeliui ir bet kuriam kito bundle'ui
(nes imame max per visus likusio plano langelius ir bundle'us). Vadinasi
`should_return` sąlyga (`level < E_home[current] + e_next_bundle + reserve`) yra
provably `False` — jokio sprendimo neprarandame skip'indami. `delta` sugeria
≤1-langelio poziciją paklaidą.

**Saugojimas:** kiekvieno sortie arm log'inamas atskirai (per-sortie sąrašas),
nepainiojant kuris galioja kada — indeksuota sortie numeriu; log per sortie
(A/B observability, §10).

### 7c. S3_RTH grįžimo kelias — parent-pointer route
`agent.py:321-323` (`ret = motion.plan(self.pose, self.base, CRUISE)`) keičiamas į:
sekti parent pointer'ius nuo `cell(self.pose)` bazės link → polyline → motion model
(§6.1). Tai **pagrindinis obstacle-boxing fix grįžimo kelyje**: kelias eina aplink
kliūtis pagal occupancy, ne per jas. Touch point: S3_RTH entry `_apply_transition`
(`agent.py:321-323`), už map flago.

### 7d. Resume transit (bazė → gilus leg) — tas pats map atbulai
`_resume_transit` (`agent.py:390-398`) naudoja tą patį map: parent-pointer kelias
nuo bazės iki resume entry langelio (map yra grįžimo kaina, tad atbulinis kelias =
bazė→entry). **Santykis su FIX-B1:** kai map flag ON, map PAKEIČIA `route_transit`
šiame kelyje. `transit_free_space` (`config.py:120`, `simulation_engine.py:223-231`)
lieka **nepriklausomas flag'as** — švariam A/B (galima lyginti: styga vs FIX-B1
visibility vs energy map). Touch point: `_resume_transit` + `_transit_planner`
injekcija (`simulation_engine.py:223-231`).

### 7e. Skip-and-flag, NE tylus drop — RUNTIME skip-on-stall (Stage 4, shipped)
> **PATIKSLINTA (Stage 4 M0 matavimas):** plan-time `E_home = ∞` kriterijus
> **FALSIFIKUOTAS** — nenaudojamas. Šis skirsnis aprašo TIKRĄ shipped mechanizmą.

Ankstesnis pasiūlymas praleisti coverage leg'ą jau **plan-time**, jei jo entry
langelio `E_home = ∞`, buvo atmestas: `E_home = ∞` NĖRA pasiekiamumo testas. M0
matavimas (50 replikacijų) parodė 432/5520 ∞-entry strip'ų **sėkmingai** įvykdomus
per S_OBS sub-cell koridorius (20 m raudonas langelis nereiškia nepasiekiamo
strip'o). Plan-time skip būtų pavertęs 46 sėkmes į PARTIAL ir numušęs sample
ceiling'ą **92% → 0%**.

Vietoje to shipped mechanizmas yra **runtime skip-on-stall**: empirinis
nepasiekiamumo įrodymas yra pats FIX-B4 `StallDetector` biudžeto pataikymas
(5 no-progress swap'ai, `simulation_engine.py:99, 513`). Kai budget hit ∧
`safety.stall_skip`: engine kviečia `agent.skip_stuck_leg()` (`agent.py:469`) →
strip'as įrašomas į `MissionResult.skipped_legs` ((agent_id, leg) poros,
`core_types.py:337`, `simulation_engine.py:480, 570`) → terminal outcome tampa
`Outcome.MISSION_PARTIAL` (`enums.py:155`, `simulation_engine.py:659`). NE tylus
drop — coverage gap log'inamas ir raportuojamas. Gate: `safety.stall_skip`
(default OFF, reikalauja `safety.stall_detector`, `config.py:204, 729-730`);
flag-off metu `skip_stuck_leg` niekada nekviečiamas → `skipped_legs` visada tuščias
→ byte-identiška. „Skip-and-flag, ne tylus drop" dvasia išlieka — tik trigger'is
yra runtime stall, ne plan-time reachability.

---

## 9. Battery-zone nets demotion (sprendimas 8) — MERGED (B1)

> **Statusas: ĮGYVENDINTA** (`rth.energy_map.zone_demotion`, PR #38). Sprendimas
> priimtas; žemiau — galutinis dizainas su B2 floor matavimo rezultatu.

**B1 — CRITICAL branch pašalinimas.** Kai map sprendžia (`zone_demotion = true`,
reikalauja `decide`), statinio battery-zone net'o `critical_battery` guard-šaka
**pašalinama** (`state_machine.py:113`, `_zone_demotion` sąlyga), paliekant
`terminal_battery` (`< battery_zones.critical`, `:115-116`) kaip vienintelį
saugiklį. **Nauja konstanta NEĮVEDAMA, `battery_zones` vertės NEKEIČIAMOS.**

Būtina pataisa: `critical_battery` guard suveikia ne ties 0.20, o **ties
`nominal = 0.40`** — CRITICAL zona yra `[critical, nominal) = [0.20, 0.40)`
(`battery.py:44-50`, `config/default.yaml:215-216`), tad guard'as kerta grįžimą
vos `f` nukrenta žemiau **0.40** (`run_regime_calculator.py:48-52`). 0.20 yra tik
TERMINAL pradžia, NE grįžimo taškas. Todėl demotion pašalina **0.40** statinį
net'ą, o TERMINAL failsafe lieka ties **0.20**.

**Kodėl „~0.15 terminal" atmestas.** `battery_zones.critical` nuleidimas iki 0.15
grįžimo laikui yra **no-op**: guard'as vis tiek suveikia ties 0.40, o nuleidimas
tik praplečia CRITICAL zoną iki `[0.15, 0.40)`. Grįžimą valdo šakos pašalinimas,
ne slenksčio vertė — todėl jokio 0.15.

**B2 — TERMINAL floor sweep (matavimas, pasirinktas floor 0.10).** Po demotion'o
lieka klausimas, koks TERMINAL failsafe floor (`battery_zones.critical`) netrukdo
map'ui. Sweep'o rezultatas (1 km²):
- floor **0.20**: statinis trupmeninis net'as **perima ~47% grįžimų** iš willing
  distance-aware map'o (14 `terminal_battery` vs 16 `rth_energy`);
- floor **0.10**: override'as dingsta — `terminal_battery → 0`, map vienas valdo
  kiekvieną grįžimą, arrival-reserve min 0.087;
- **0.10 ≡ 0.05** 1 km² byte-identiškai (žemiau ~0.12 map visada grįžta pirmas),
  tad floor'o vertė yra **scale-axis klausimas**, ne 1 km² klausimas;
- swap'ai kvantuoti ties 5, ir floor jų **nepajudina** — raportuojamas negatyvas.

**Pasirinktas Stage-5 arm 4 floor: 0.10** — palieka nepriklausomą ~5 pp backstop'ą
virš `reserve_frac`. Taikoma **in-config** tam eksperimentui, NE keičiant
`default.yaml`.

Reason attribution (`state_history.reason_out`, `state_history.py:25`) su map ON
tada rodo **`rth_energy` dominuojant**, o `critical_battery` nukrinta į 0 — tai pat
matuojamas A/B observable (§11).

---

## 10. Gating & rollout (sprendimas 9)

**Flag'ai, default OFF.** Siūloma nauja config sekcija arba `RTHConfig` plėtinys
(`config.py` schema po `RTHConfig`, `:196-198`), pvz. `rth.energy_map` su
`enabled: false` + per-stage sub-flag'ai. Byte-identity strategija identiška
esamiems opt-in flag'ams (`ferry_free_space`, `transit_free_space`,
`sensor_power_w`, `obstacle_recovery`, `stall_detector`): naudoti tą patį
**optional-key provenance-hash** triuką (raktas absent default.yaml → hash ir visi
fixture'ai nepakitę; žr. `config.py` komentarus `:93-116, 182-192, 270-280`).
Flag-off runs turi būti **byte-identiški** pre-change.

**Staged planas (kiekviena stadija: flag + testai + DoD):**

| Stadija | Kas | Per-stage testai | DoD |
|---|---|---|---|
| S1 Map builder | occupancy fraction + Dijkstra + `E_home`/parent; grynas skaičiavimas, jokio FSM ryšio | unit: žinomo grid'o `E_home` teisingas; laisvo grid'o `E_home = dist×18.333`; red/yellow klasifikacija; parent grandinė pasiekia bazę | builder žalias; flag-off nekviečiamas |
| S2 decide + arming | 7a + 7b už flago | `should_return` per map == analitinis; arm bound (virš arm niekada negrąžina); flag-off byte-identity | A/B decide veikia; senas 5 s kelias išlieka |
| S3 Return routing | 7c parent-pointer grįžimas | grįžimo kelias aplenkia kliūtį (S_OBS nekyla); flag-off == tiesi styga | boxing fix grįžime |
| S4 Resume routing | 7d atbulinis map | resume aplenkia kliūtį; `transit_free_space` nepriklausomas | livelock resume fix |
| S5 Skip-on-stall | 7e runtime skip-on-stall + `skipped_legs` (`safety.stall_skip`, reikalauja `stall_detector`) | stall budget hit → `skip_stuck_leg` → `MISSION_PARTIAL`; flag-off `skipped_legs` tuščias (byte-identity) | apskaita eksplicitiška, ne tylus drop |

Kiekviena stadija: pilnas `pytest` žalias; nauji testai naujam elgesiui;
flag-off byte-identity fixture (hash nepakitęs). Battery-zone demotion (§9)
įjungiama tik po S2 (kad `rth_energy` jau veiktų prieš nuimant `critical_battery`).

---

## 11. A/B eksperimentas (eskizas)

**Tikslas:** ar dinaminė energy-map RTH pralenkia statinį slenkstį, paired seeds.

**Keturios rankos** (task C1; ne trys) ant tapačių seed'ų:
1. **static-40% (map OFF)** — literatūros baseline: senasis 5 s laiko kelias +
   aktyvūs battery-zone nets. **PATIKSLINTA:** faktinis statinis net'as suveikia
   ties **`nominal = 0.40`** (CRITICAL guard'as, `state_machine.py:113`,
   `run_regime_calculator.py:48-52`), NE 20% — 0.20 yra tik TERMINAL pradžia.
   Dokumentuoti kaip „static 40% net", niekada „static 20%".
2. **route-only** — map maršrutizuoja grįžimą/resume (7c+7d), bet decide lieka
   statinis; CRITICAL guard'as nepaliestas.
3. **decide+route (CRITICAL intact)** — map sprendžia IR maršrutizuoja
   (7a+7b+7c+7d), bet `zone_demotion` OFF → statinis 0.40 net'as dar preemptina.
4. **decide+route + `zone_demotion`** — pilnas efektas: CRITICAL šaka pašalinta
   (§9), map vienas valdo grįžimą; TERMINAL floor 0.10 (in-config).

- **Paired seeds:** ta pati `RngFactory` visoms rankom → `rng.stream(name, rep)`
  tapati (`rng.py:47-54`); fixed-N (ne CI stopping), kad pairing išliktų tikslus.
- **Grid:** L-shape × area tiers {1,2,4,8,16 km²} (§3) × obstacle count
  (`obstacle_density_per_km2`, default 8/km², `config/default.yaml` env).

**Laukiami observables:**

| Metrika | 1. static-40% | 2. route-only | 3. decide+route (CRITICAL intact) | 4. +zone_demotion |
|---|---|---|---|---|
| Naudingas sortie langas | ~60% (1.0→0.40) | ~60% (CRITICAL intact) | ~60% (CRITICAL intact) | ~80% (1.0→0.20) |
| Demand median (swaps) | ~8 | ↓ | ↓ | ↓ |
| Success ceiling | ~92.8% | ↑ (boxing fix) | ↑ | ↑ |
| `rth_energy` count | ~0 | yra | yra | **dominuoja** |
| `critical_battery` count | dauguma | dauguma | dauguma | **~0** |

(Arm 1 skaičiai — **AUTORIAUS DIAGNOZĖ** STUDY-01; kitų rankų kryptys —
hipotezės, patikrinamos realiu paleidimu. Naudingas langas: statinis net'as
kerta ties 0.40 → ~60% (1.0→0.40); tik `zone_demotion` atveria iki ~80%
(1.0→0.20).) `rth_energy`/`critical_battery` skaičiai iš
`state_history.reason_out` (`state_history.py:25`) /
`smdp_estimator.n_transitions` (`smdp_estimator.py:49`).

---

## 12. Ko šis dizainas NEatsako (sąžiningos ribos)

1. **Coverage legs vis dar guli virš kliūčių.** Map tvarko grįžimo/transit/resume
   kelius, NE coverage strip'ų geometriją. In-strip S_OBS sidestep'ai
   (`_avoidance_plan`, `agent.py:400-406`) lieka — strip'ą per kliūtį vis tiek
   reikia praeiti. (FIX-B1 `ferry_free_space` tvarko connector'ius atskirai.)
2. **Nėra vėjo / voltage-sag simuliacijoje.** Cell-size taisyklė (§3) robustiška
   TIK baterijos-capaciteto pokyčiams (360 J/cell perskaičiuoja automatiškai).
   Vėjas ar įtampos kritimas keistų realią J/m ne per capacitetą — ne modeliuojama.
3. **Grid anisotropy.** 8-connected kvadratinis grid'as turi kryptinį artefaktą
   (diagonalės √2, bet 22.5° kryptys neidealios). Hex (§3) tai spręstų — future.
4. **Not-in-area ferry.** Skrydis už survey polygono (ferry metu leidžiamas,
   `coverage` operating_area, `visibility_router.flyable_region`) energy map'e
   traktuojamas konservatyviai (bbox-in-area red) — map neišnaudoja viso flyable
   ploto už polygono. Sąmoningas supaprastinimas.
5. **Literatūros nota (kartojama).** Cost-to-go grid'ai klasikiniai; indėlis =
   battery-normalizuota per-rep energy map, keičianti statinį RTH slenkstį,
   patikrinta paired-seed A/B — ne pats grid'as.

---

## Priedas A: šaltinių santrauka

| Teiginys | Šaltinis |
|---|---|
| Cap 360 kJ (100 Wh × 3600) | `config/default.yaml` fleet; `config.py:29, 365` |
| Reference cruise 18.333 J/m (220 W / 12 m/s) | `config/default.yaml` platforms.MULTIROTOR; `energy_model.py:106-118` |
| `reserve=0.05·cap=18 kJ` (skaič. `rth_calculator.py:80`), `check_interval_s=5 s` | `config/default.yaml` rth; `config.py:196-198` |
| Zones high/nominal/critical = 0.75/0.40/0.20; `critical_battery` guard suveikia ties nominal=0.40 (CRITICAL juosta [0.20,0.40)), NE vien reporting bin | `config.py:155-159`; `battery.py:44-50`; `run_regime_calculator.py:48-52` |
| Guard tvarka: rth_energy→critical_battery (<0.40)→terminal (<0.20) | `state_machine.py:109-116` (`:111-112`, `:113-114`, `:115-116`) |
| `should_return = level < e_next + return + reserve`; ×1.5 fudge | `rth_calculator.py:63-82` (`:68-69`) |
| RTH check kas 5 s, tik S2/S_FERRY | `agent.py:225-227` |
| `lookahead` bundle (leg + connector + camera) | `agent.py:411-429` |
| S3_RTH tiesi styga | `agent.py:321-323` |
| resume-transit; 15 m sidestep; boxed-in→S3 (budget 6) | `agent.py:390-398, 400-406, 50, 342-344` |
| path_clear „~63% of mission runtime" | `environment_map.py:157-158, 186-193` |
| `occupancy_grid` + `GridFrame` (reuse) | `environment_map.py:195-205, 25-37` |
| `buffered_obstacles`; `segment_in_obstacle` (RAW) | `environment_map.py:72-80, 120-136` |
| `distance_energy` (Dijkstra edge cost) | `energy_model.py:106-118` |
| Motion smoothing seam (Holonomic/Dubins) | `motion_model.py:143-167, 129-134` |
| polyline→Path chain pavyzdys | `visibility_router.py:207-225` |
| FIX-B1 route_transit; `transit_free_space` | `visibility_router.py:228-275`; `config.py:120`; `simulation_engine.py:223-231` |
| Per-rep build seam (env+base+obstacles static) | `simulation_engine.py:166, 175, 181-185, 190, 203, 295-298, 309-313` |
| `MissionResult.stalled_agents` (skip-flag precedentas) | `core_types.py:330`; `simulation_engine.py:440` |
| `reason_out`; `n_transitions` (A/B counts) | `state_history.py:25`; `smdp_estimator.py:49` |
| RngFactory.stream paired seeds | `rng.py:47-54, 21` |
| L-shape bbox 1155 m, fill 0.75; grid dims | `data/areas/shapes/l_shape.geojson` (apskaičiuota) |
| Grid-cost, build-laikai, yellow-penalty derivacija | **ESTIMATE** — formulės pateiktos §3–§6 |
| STUDY-01 92.8%/36-500; FIX-B1 347–791 s/rep | **AUTORIAUS DIAGNOZĖ** (ne repo artefaktas; `docs/reports/` tuščias) |

## Priedas B: roadmap įrašas (istorinis — jau įgyvendinta)

Roadmap dabar gyvena `TODO.md` (`docs/thesis_roadmap.md` ištrintas). EM-01 jau
merged — žr. `TODO.md` B1/C1 eilutes. Istorinis siūlytas įrašas (dabar realizuotas):

> | EM-01 | Energy cost-to-go map: per-rep Dijkstra `E_home`+parent; obstacle-aware
> RTH decide (battery-quantized cadence) + return/resume routing; static-RTH
> demoted TERMINAL-only. Staged, flag'uota, byte-identity. A/B: static-40% vs
> map (keturios rankos). | **Code / Plan** | šis pasiūlymas | Merged; A/B verdiktas
> (success ceiling, sortie depth, rth_energy dominance) honest read-out |
