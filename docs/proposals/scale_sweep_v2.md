# Scale sweep v2 — dizaino pasiūlymas (sprendimui su vadovu)

> **Statusas:** PASIŪLYMAS — eksperimentas nevykdomas, kol nepasirinktas scope
> (žr. §5 sprendimo taškus). Dokumentas savarankiškas: visi skaičiai turi šaltinį
> (failas:eilutė, `runs/<run>/run.json` arba įvardintas paleidimas); įverčiai be
> tiesioginio šaltinio pažymėti **ESTIMATE** su formule ir prielaidomis.
>
> Data: 2026-07-13 · bazinis commit: `ef15d41` (main)

---

## 0. Prerequisites ir duomenų higiena

| # | Prerequisite | Statusas 2026-07-13 |
|---|---|---|
| D1 | S5 formalus read-out (`docs/reports/s5_shipped_readout.md`, `s5_clean_readout.md`, NOW-03) | **NĖRA main'e** — `docs/` kataloge tik `thesis_roadmap.md`. §1.3 pastraipa remiasi tiesiogiai run artefaktais (`runs/shape_sweep_clean_postfix/`), ne read-out dokumentu. Formalus NOW-03 verdiktas — atvira priklausomybė. |
| D2 | `run_spare_sizing --demand-mode` (B=∞ demand matavimas) | **YRA main'e** — PR #28, commit `ef15d41`; D fiksuojamas bendras ir per-drone (`spare_sizing.py:279-280`, `DemandRecord.demand` + `per_drone_swaps`). |
| D3 | Analitinis N\* skaičiuotuvas | **VEIKIA** — `run_fleet_sizing_analyzer` patikrintas realiu paleidimu 2026-07-13 (git `ef15d41`), žr. §4.1. Pastaba: roadmap'o įrašas FIX-01 („analyzer broken") pasenęs — importo klaidos nebėra. |
| D4 | Ploto tier'ų geometrija | **YRA** — `generate_shapes.py` turi `--target-area-m2` (normalizacija į tikslų plotą: `generate_shapes.py:144-150`), t. y. scaled formų šeimoms naujo kodo nereikia. |

**Duomenų higiena (privaloma skaitant senesnius juodraščius).** 2026-07-05 rastas ir
pataisytas launch-site RNG defektas (commit `a0871b6`, PR #18): iki tol kiekviena MC
replikacija persirinkdavo launch pad'ą, ir tai užteršė visų optimizer-sited variantų
dispersiją. **Visi skaičiai iš iki-pataisos `shape_sweep_clean` run'o yra negaliojantys** —
tarp jų anksčiau cituoti „TGC vs classic +0.31", „optimizer wastes +13.7 kJ",
„H2 corr −0.137". Šiame dokumente naudojami tik šie šaltiniai:

- `runs/shape_sweep_clean_postfix/` — clean grid'as, git `1214148` (po RNG pataisos
  ir po path_clear spartinimo), 2026-07-12;
- `runs/shape_sweep_shipped/` — shipped grid'as, git `e9c40e2` (po RNG pataisos), 2026-07-09;
- kodo faktai (failas:eilutė) ir įvardinti analitiniai paleidimai.

---

## 1. Motyvacija

### 1.1 53 h incidentas: dabartinė scale konfigūracija astronomiškai brangi

Bandymas paleisti pilną fleet-scale grid'ą Azure (2026-07,
`run_scale_tiers --budget full --mode shipped`) po **53 valandų buvo ties 0/50
užbaigtų tier'ų** (autoriaus Azure sesija; py-spy diagnozė identifikavo
`path_clear` hot spot'ą, vėliau pataisytą PR #26, commit `1214148`). Priežastis —
konfigūracijos sandauga:

- `--budget full` = 50 fleet-size tier'ų, n = 2, 4, …, 100
  (`run_scale_tiers.py:69`, `_BUDGET_GRIDS`);
- adaptyvus MC iki `mc.n_max = 1000` replikacijų tier'ui (`config/default.yaml:244`),
  minimum 30 (`config/default.yaml:245`);
- `--mode shipped` — su kliūtimis, brangiausias per-mission režimas.

Net po path_clear pataisos (~2.3× per misiją) tokia sandauga lieka nepakeliama —
problema ne implementacijos, o **dizaino**: grid'as be analitinio prior'o apie tai,
kur n ašyje apskritai yra informacijos. Šis pasiūlymas keičia dizainą, ne kodą.

### 1.2 1 km² caveat: mažas plotas gali nediferencijuoti algoritmų

Autoriaus įžvalga: visas S5 grid'as vyksta ant 1 km² lygiaplotių formų, kur vienas
dronas su 100 Wh baterija (`config/default.yaml:27`) padengia didelę ploto dalį per
vieną sortie. Tą patvirtina S5 artefaktai:

- **Regime žymos:** `runs/shape_sweep_clean_postfix/summary.md` lentelėje ties n ≥ 4
  dauguma langelių FUEL-SURPLUS (išimtis — pinwheel, kuri BATTERY-LIMITED visame
  n = 2..6). Battery-limited / multi-sortie režimas, kuriame veikia S_SWAP eilė ir
  RTH ekonomika, 1 km² gride beveik neaktyvuotas.
- **Swap ašis nejautri:** paired kontrastas `tgc_basic − kmeans` metrikai `swaps`
  yra **lygiai 0 visose 45 clean langeliuose** (`runs/shape_sweep_clean_postfix/contrasts.csv`,
  metric=swaps agregatas). Decomposition algoritmų skirtumas swap paklausai 1 km²
  masteliu neišmatuojamas iš principo — visi variantai telpa į tą pačią sortie
  struktūrą.

Jei formos / decomposition efektai priklauso nuo mastelio, 1 km² išvados gali būti
tik apatinis taškas, o ne atsakymas.

### 1.3 Ką jau žinome iš S5 (preliminarus skaitymas iš run artefaktų)

Formalus NOW-03 read-out dar nesumerge'intas (§0 D1), todėl žemiau — tiesioginis
skaitymas iš `runs/shape_sweep_clean_postfix/contrasts.csv` (45 langelių, 9 formos ×
n = 2..6, po 20 paired replikacijų). TGC prieš classic_voronoi yra robustiškas
visose „lower-is-better" metrikose: total_energy vidutiniškai **−99.5 kJ** (TGC
geresnis 37 iš 41 nenulinių langelių), makespan **−1293 s** (38/41), swaps **−1.29**,
energy_imbalance **−0.84**. TGC prieš kmeans — praktiškai lygiaverčiai: total_energy
vidurkis **−3.2 kJ** TGC naudai, bet ženklas mišrus (TGC geresnis tik 11/40
nenulinių langelių — vidurkį lemia keli dideli laimėjimai), SMDP efficiency
**−0.034** kmeans naudai (33/40), o
ant neutralaus launch pad'o (Problem-B kontrastas `tgc_naive_launch −
kmeans_naive_launch`) efficiency skirtumo vidurkis **−0.007** su mišriu ženklu
(17 teigiamų / 21 neigiamas) — skirtumas ties matavimo triukšmo riba. Scoped null
patvirtintas tiksliai: `weighted_voronoi − tgc_basic ≡ 0` visur
(`summary.md`: null_max_abs = 0, null_all_exact = True). Klausimas, kurio šie
duomenys atsakyti negali: **ar šis paveikslas išlieka augant plotui.**

---

## 2. Hipotezės

**H-A (formos efekto silpnėjimas).** Formos efektas silpnėja augant plotui: nuo tam
tikro ploto dominuoja dydis, ne forma. Analitinis argumentas (ESTIMATE, formulė):
coverage darbas auga ∝ A (strip ilgis = A/swath), o formos nulemti dėmenys —
tranzitai, konektoriai, perimetro geometrija — auga ne greičiau kaip ∝ √A
(charakteringas linijinis matmuo). Santykinis formos indėlis tada ~ √A/A = **1/√A**:
tarp 1 km² ir 16 km² tikėtinas ~4× susitraukimas. Falsifikuojama prognozė: per-tier
kontrastų (pvz., TGC − classic) santykinis dydis mažėja su A. Jei NEmažėja —
falsifikacija irgi yra rezultatas (formos efektas mastelio-invariantas), ir tai net
stipresnė žinia tezei.

**H-B (swap paklausos skalė).** Swap-pack paklausa D skaluojasi ~ E_cover/B_usable.
Analitinis prior'as jau yra kode: `fleet_sizing.total_sorties` =
E_cover / coverage_budget_per_sortie (`fleet_sizing.py:201-205`), o fleet-wide swap
skaičius = total_sorties_int − n_active (`fleet_sizing.py:234-236`). Empirinė pusė —
demand-mode ekvivalentumas success(k, B) ⟺ D_k ≤ B (`spare_sizing.py:258-260`):
išmatavus D po neribotu pool'u, visa success-vs-B kreivė atkuriama post hoc, todėl
**B (pool dydžio) dimensijos grid'e apskritai nereikia**. Falsifikuojama prognozė:
išmatuotas D auga ~tiesiškai su A fiksuotam n ir sutampa su analitiniu prior'u
(paklaida — obstacle detour'ai, kurių analitinė formulė yra floor).

---

## 3. Dizainas

### 3.1 Ašys

| Ašis | Reikšmės | Pagrindimas |
|---|---|---|
| Forma | 9 esamos lygiaplotės formos (`data/areas/shapes/`: square, rect_2_1, rect_4_1, rect_8_1, disk, l_shape, star_5, pinwheel, c_shape) | S5 tęstinumas; kiekvienam tier'ui šeima regeneruojama `generate_shapes --target-area-m2` (formos deskriptoriai — solidity, isoperimetric — masteliui invariantiški, todėl H-A skaitymas švarus) |
| Plotas | **5 tier'ai geometrine ×2 progresija: 1, 2, 4, 8, 16 km²** (žr. §5 K1) | 1 km² = S5 inkaras (tiesioginis palyginamumas); ×2 žingsnis duoda tolygų log-tinklą H-A regresijai; 16 km² jau giliai multi-sortie režime (ESTIMATE §4.2: ~38 sorties vienam dronui ties n=1) |
| Fleet dydis | n ∈ [2 .. 2·N\*(shape, area)] iš analitinio prior'o (§3.2); tankis — sprendimo taškas §5 K2 | Vietoje aklo n = 2..100: grid'as dengia [under-provisioned .. 2× virš knee] diapazoną, kur ir yra visa informacija |
| Baterija | **FIKSUOTA** 100 Wh (`config/default.yaml:27`) | B kaip pool dimensija pašalinta per demand-mode (§2 H-B); battery capacity kaip ašis — out of scope (S5 sprendimas galioja toliau) |

### 3.2 Analitinis N\* pre-step (pigus, be simuliacijos)

Kiekvienai (forma, tier) kombinacijai prieš MC paleidžiamas grynas analitinis
skaičiavimas: `fleet_sizing.sweep()` (`fleet_sizing.py:239-302`) ant clean planning
layer'io — Pareto lentelė + knee (`run_fleet_sizing_analyzer.py:119-129`,
`_find_knee`, 5 % marginal taisyklė). Kaina — sekundės vienai kombinacijai
(planning fazės sudaro ~1 % misijos laiko: `runs/shape_sweep_shipped/profiling.md`,
dt_loop = 99.0 %).

**Knee vs saturacija (svarbi subtilybė n-grid ribai).** Realus paleidimas
(2026-07-13, git `ef15d41`, default config — example_area, navigable 2.53 km²,
100 Wh): total sorties 5.96 → 6, knee N\* = 4, BET ties N = 6 dar −28 min šuolis,
nes 6 sorties pasidalija lygiai (visa Pareto lentelė §4.1 kalibracijai). 5 %
taisyklė gali „užsifiksuoti" prieš vėlesnį lygaus dalijimosi šuolį, todėl siūloma
n-grid viršūnė **2·max(knee, sorties_int)** — žr. §5 K2.

### 3.3 Protokolas (S5 paveldas, nekeičiamas)

- **7 variantai per langelį:** 4 decomposition peers (tgc_basic, weighted_voronoi,
  classic_voronoi, kmeans) + 3 naive-launch twins (tgc/classic/kmeans_naive_launch,
  `run_shape_sweep.py:134-152`). Metodologinis tęstinumas su S5; scope taupomas per
  tier'us/formas, ne per variantus (weighted_voronoi kainuoja, bet dokumentuoja
  scoped null kiekviename mastelyje).
- **Paired seeds, fixed N:** `RngFactory.stream(name, replication)` — gryna
  (master_seed, name, replication) funkcija; fixed N (ne adaptive stopping) išsaugo
  tikslų porų palyginimą tarp variantų (`run_shape_sweep.py:35-41`).
- **Clean PRIMARY** (obstacle_density = 0 — grynas formos+mastelio efektas,
  atitinka analitines žymas tiksliai) **+ shipped spot-check** keliuose tier'uose
  (kliūčių tankis fiksuotas per km² — `config/default.yaml:151`: 8/km², t. y.
  16 km² tier'e ~128 kliūčių; robustness, ne primary).
- **Metrikos per langelį** (mean ± 95 % CI per paired replikacijas): SMDP
  efficiency (headline), total_energy, makespan, executed energy imbalance
  (max/mean), swap count (`run_shape_sweep.py:42-47`). **D (swap-pack demand):**
  sweep'o variklyje pool neribotas, todėl swap count ≡ D (bendras); per-drone D
  pasiskirstymui — taikinis demand-mode paleidimas `run_spare_sizing --demand-mode`
  pasirinktuose langeliuose (`DemandRecord.per_drone_swaps`, `spare_sizing.py:279-280`),
  kaina §4.4.
- **Runner:** `run_shape_sweep` išplėtimas ploto/tier ašimi — atskira ENG užduotis
  po scope patvirtinimo (šis dokumentas kodo nekeičia). Esami seam'ai: per-cell
  config konstrukcija (`build_cell_cfg`), unikalūs run katalogai, `--jobs`
  paralelizmas, crash-safe incremental log — visi jau main'e.

---

## 4. Biudžetas

### 4.1 Kalibracija (realūs matavimai)

| Šaltinis | Faktas |
|---|---|
| `runs/shape_sweep_clean_postfix/run.json` | clean full: 45 langelių × 7 variantai × N=20 = 6 300 misijų per wall 11 466 s (`--jobs 3`, 4-core Azure Linux, git `1214148` — path_clear 2.3× spartinimas JAU įskaičiuotas) → **1.82 s/misija** (1 km², amortizuota) |
| `runs/shape_sweep_shipped/run.json` | shipped full: 18 langelių × 7 × N=100 = 12 600 misijų per wall 55 137 s (`--jobs 4`, git `e9c40e2` — PRE-path_clear) → 4.38 s/misija; **ESTIMATE po pataisos: 4.38/2.3 ≈ 1.9 s/misija** (1 km²) |
| `runs/shape_sweep_shipped/profiling.md` | dt_loop = 99.0 % misijos laiko; visos planning fazės ~1 % → planavimo kaštas biudžete ignoruojamas pagrįstai |
| Analitinis paleidimas 2026-07-13 (git `ef15d41`, default config) | usable battery 342 000 J; blended coverage kaina 1 800 208 J / 52 205 m = **34.5 J/m**; per-sortie overhead 40 116 J, coverage budget 301 884 J; knee N\*=4, sorties 6 |

### 4.2 Kaštų modelis (ESTIMATE — visos formulės ir prielaidos)

**T(scope) = Σ per langelius [ reps × 7 variantai × t₁ × A ]**, kur t₁ = 1.82 s
(clean) / 1.9 s (shipped, ESTIMATE), A — tier'o plotas km².

Prielaidos:
1. **Misijos wall kaina ∝ A.** Makespan ∝ A/n, o kiekvieno žingsnio kaina ∝ n
   (n dronų per dt) → sandauga ∝ A, ~nepriklausoma nuo n. Rizika: SafetyMonitor
   porų tikrinimas O(n²) (autoriaus py-spy diagnozė 2026-07, ~31 % dt_loop; ne repo
   artefaktas) — dideliems n įvertis optimistinis.
2. **Paralelizmo amortizacija kaip kalibracijos run'uose** (jobs=3–4 ant 4 branduolių;
   straggler tail ignoruojamas).
3. **Shipped ∝ A tik apytikriai:** kliūčių skaičius auga ∝ A (fiksuotas tankis/km²),
   path_clear kaina auga su kliūčių skaičiumi → shipped įverčiai dideliems tier'ams
   optimistiniai. Dar viena priežastis shipped laikyti spot-check'u.
4. **N\*(A) ESTIMATE** (tiksles reikšmes duos §3.2 pre-step):
   E_cover(A) ≈ [A·10⁶/50 + (√(A·10⁶) − 50)] m × 34.5 J/m; sorties(A) =
   E_cover/301 884 J (overhead ~konst. 40 kJ; tranzitas auga ∝ √A, tad floor):

   | A (km²) | E_cover (MJ) | sorties | N\*_sat = ⌈sorties⌉ | n-grid [2..2N\*] dydis (kas 1) |
   |---:|---:|---:|---:|---:|
   | 1 | 0.72 | 2.4 | 3 | 5 |
   | 2 | 1.43 | 4.7 | 5 | 9 |
   | 4 | 2.83 | 9.4 | 10 | 19 |
   | 8 | 5.62 | 18.6 | 19 | 37 |
   | 16 | 11.18 | 37.0 | 38 | 75 |

### 4.3 Scope variantai (vadovo pasirinkimui)

| | **A — minimalus** | **B — vidutinis (siūlomas)** | **C — pilnas (NErekomenduojamas)** |
|---|---|---|---|
| Tier'ai | 1, 4, 16 km² (×4) | 1, 2, 4, 8, 16 km² (×2) | 1, 2, 4, 8, 16 km² |
| Formos | 4 (square, rect_4_1, star_5, c_shape — convex/pailga/spygliuota/įgaubta) | 6 (+ rect_2_1 ir (rect_8_1 arba pinwheel; disk/l_shape metami)) | visos 9 |
| n-taškai / tier | 4: {2, N\*, ⌈1.5N\*⌉, 2N\*} | 4: {2, N\*, ⌈1.5N\*⌉, 2N\*} | pilnas [2..2N\*] kas 1 |
| Reps (clean) | N=10 | N=20 (S5 full paritetas) | N=20 |
| Shipped spot-check | — | tier'ai {1, 4}, n ∈ {2, N\*}, N=20 | + pilnas shipped (~2× clean) |
| Langelių (clean) | 48 | 120 | 1 305 |
| Misijų | 3 360 | 16 800 + 3 360 shipped | 182 700 |
| **Wall ESTIMATE (4-core Azure)** | **≈ 12 h (~12 VM-h)** | **≈ 53 h clean + 4.4 h shipped ≈ 57 h (~2.4 paros)** | **≈ 1 016 h ≈ 42 paros** |

Skaičiavimo pavyzdys (B, clean): 24 langeliai/tier × 140 misijų × 1.82 s ×
Σ(1+2+4+8+16) = 6 115 s × 31 = 189 571 s ≈ 52.7 h. (C variantas — tas pats 53 h
incidento paveikslas, tik dabar matomas IŠ ANKSTO: būtent tam ir yra ši lentelė.)

### 4.4 Papildomi kaštai

- **Analitinis N\* pre-step:** sekundės × (formos × tier'ai) — nereikšminga.
- **Demand-mode per-drone gylis (H-B):** `run_spare_sizing --demand-mode --reps 500`
  vienam (forma, tier, n) langeliui ≈ 500 × 1.82 s × A: 1 km² ~15 min, 4 km² ~1 h,
  16 km² ~4 h. Siūloma: square + c_shape, po vieną n=N\* langelį kiekviename
  tier'e → B scope ≈ 2 × (0.25+0.5+1+2+4) h ≈ **15.5 h** (ESTIMATE, ta pati ∝A
  prielaida). Įtraukimas — §5 K4.

---

## 5. Vadovo sprendimo taškai

**K1 — Ploto tier'ai.** Siūloma 5 tier'ų ×2 progresija iki 16 km². Alternatyvos:
4 tier'ai iki 8 km² (pigiau, bet H-A svertas 1/√A tesusitraukia ~2.8×) arba ×4
progresija {1, 4, 16} (pigiausia, 3 taškai — minimalus log-regresijos pagrindas).
*Kiek tier'ų ir koks maksimalus plotas?*

**K2 — n-grid riba ir tankis.** Siūloma viršūnė 2·max(knee, sorties_int) (žr. §3.2
knee-vs-saturacijos subtilybę) ir 4 log-taškai vietoje „kas 1" (pilnas žingsnis ties
16 km² duoda 75 n-reikšmes — C varianto sprogimo šaltinis). *Ar 2N\* viršūnė ir
4 taškai priimtini, ar reikia tankesnio n tinklo kuriame nors tier'e?*

**K3 — Replikacijos.** Siūloma fixed N=20 clean (S5 full paritetas; pairing
išsaugotas, CI raportuojamas, adaptive stopping NENAUDOJAMAS — jis laužytų paired
seeds protokolą, `run_shape_sweep.py:35-41`). Clean režime liekamoji dispersija —
tik launch-site sampling, tad N=10 gali pakakti (A variantas). *N=20 ar N=10?*

**K4 — Clean vs shipped proporcija.** Siūloma clean PRIMARY + shipped spot-check
tier'uose {1, 4} (B variantas). Shipped ties 16 km² (~128 kliūčių) — ir brangiausias,
ir metodologiškai triukšmingiausias. *Ar spot-check apimtis pakankama robustness
teiginiui, ir ar įtraukti §4.4 demand-mode gylį (~15.5 h)?*

---

## 6. Ko šis dizainas NEatsako (sąžiningos ribos)

1. **Battery-weighting indėlis.** Homogeniškas full-battery parkas išlaiko
   `weighted_voronoi ≡ tgc_basic` nulį KONSTRUKCIŠKAI bet kokiame mastelyje (lygios
   baterijų frakcijos → identiška particija; redistribution aktyvuojasi tik per
   failure, o λ=0). Scale sweep'as šio nulio nepajudins — battery-weighting
   vertinimas priklauso diverged-battery režimui (future work, NOW-04 framing
   sprendimas). Variantas gride paliekamas tik nulio dokumentavimui.
2. **Launch ašis.** S5 clean duomenyse launch efektas mišraus ženklo
   (`tgc_basic − tgc_naive_launch` efficiency vidurkis +0.09, bet 17/45 teigiamų
   prieš 28/45 neigiamų langelių — honest null kandidatas). Scale sweep'as launch
   ašį matuos pakeliui (naive twins), bet dedikuotas launch tyrimas (Variantas C)
   lieka atskiras klausimas.
3. **Vėjas ir dinaminės kliūtys** — nesimuliuojama (modelio ribos, ne šio dizaino).
4. **λ > 0 (gedimai)** — autoriaus sprendimu ne šio tyrimo ašis; D matuojamas
   deterministinės paklausos prasme.
5. **Statistinė galia:** fixed N su CI raportavimu — sąmoningas kompromisas dėl
   pairing; ribiniai kontrastai gali likti neišspręsti (tada raportuojami kaip
   neišspręsti, ne force-confirmuojami).
6. **Analitiniai prior'ai yra floor'ai** (obstacle detour'ai nepriskaičiuoti) —
   N\* gali būti pavertintas žemyn shipped režime; pre-step skaičiuojamas clean
   sluoksniu sąmoningai.

---

## Priedas: šaltinių santrauka

| Teiginys | Šaltinis |
|---|---|
| 50 tier'ų full grid'as (n=2..100 kas 2) | `run_scale_tiers.py:69` |
| mc.n_max=1000, n_min=30 | `config/default.yaml:244-245` |
| 53 h / 0 tier'ų; py-spy path_clear; SafetyMonitor ~31 % O(n²) | autoriaus Azure sesija 2026-07 (ne repo artefaktas); path_clear pataisa PR #26 = `1214148` |
| 1.82 s/misija clean kalibracija | `runs/shape_sweep_clean_postfix/run.json` (wall_time_s=11 466.487; 45×7×20 misijų) |
| 4.38 s/misija shipped (pre-fix) | `runs/shape_sweep_shipped/run.json` (wall_time_s=55 137.134; 18×7×100) |
| dt_loop 99.0 % | `runs/shape_sweep_shipped/profiling.md` |
| TGC−classic: −99.5 kJ, −1293 s ir kt.; TGC−kmeans: −3.2 kJ / −0.034; Problem-B −0.007; weighted nulis; swaps≡0 | `runs/shape_sweep_clean_postfix/contrasts.csv` (per-contrast agregatai per 45 langelius) ir `summary.md` |
| Regime žymos (FUEL-SURPLUS dominuoja n≥4) | `runs/shape_sweep_clean_postfix/summary.md` per-cell lentelė |
| 34.5 J/m, 342 kJ usable, overhead 40.1 kJ, knee N\*=4 / sorties 6 | `run_fleet_sizing_analyzer` paleidimas 2026-07-13, git `ef15d41`, default config |
| Knee taisyklė; sorties/swap formulės; demand ekvivalentumas | `run_fleet_sizing_analyzer.py:119-129`; `fleet_sizing.py:201-205, 234-236`; `spare_sizing.py:258-260, 279-280` |
| 7 variantų protokolas, fixed-N pairing, metrikos | `run_shape_sweep.py:35-47, 134-152` |
| Formų normalizacija į taikinį plotą | `generate_shapes.py:144-150` (`--target-area-m2`) |
| E_cover(A), sorties(A), N\*(A), visi wall laikai §4.2–4.4 | **ESTIMATE** — formulės ir prielaidos pateiktos vietoje |
