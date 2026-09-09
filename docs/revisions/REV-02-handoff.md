# REV-02 perdavimas: coherent Lloyd perskirstymas ir individualus RTH

## Autoritetinga gyva būsena ir plano revizija

`execution.agent.Agent` yra vienintelis autoritetas BO fizinei būsenai:
`pose` (įskaitant heading), `battery`, `energy_consumed_j`, `flown_m`,
coverage raster observer ir `PhotoTracker` niekada nėra atkuriami keičiant
planą. Coherent AGL autoritetingai laikomas
`agent._coherent.altitude_m`; `Agent.view()` perduoda jį kartu su konkrečiu
`agent.base` planavimo sluoksniui.

`Repartitioner.attempt()` sudaro visų BO kandidatus nekeičiant agentų.
Kiekvienam jis iškviečia `Agent.prepare_retask(...)`; coherent šaka tikrina
maršruto tęstinumą iš tikros pozos, likusį vertikalų kilimą, kiekvieno
produktyvaus leg RTH ir pirmą darbo/RTH energijos paketą. Tik kai kiekvienas
kandidatas priimamas, `SimulationEngine._run_repartition()` kviečia
`Agent.commit_retask(...)`. `plan_revision` prasideda nuo 0 ir didėja tik
priimtam planui. Atmetimas `candidate_rejected` nekeičia partition, planų ar
BO būsenos.

`commit_retask` išvalo tik seno plano eilę, coverage leg indeksą ir coherent
plano/RTH cache. Nereikia ir negalima atstatyti baterijos, AGL, pozos,
energijos, atstumo, rasterio ar fotografavimo istorijos. Jei BO jau ore,
`CoherentFlight.transition_legs()` naudoja `retask_transit_legs()`: nuo esamo
AGL atliekamas tik likęs kilimas, o ne naujas kilimas nuo žemės.

## RTH commitment ir vykdymo kelias

`CoherentFlight.prepare_return()` yra RTH commitment vieta. Ji kviečia
`RthCalculator.return_plan(agent.pose, base=agent.base,
altitude_m=actual_agl)`; tas pats calculator sprendžia obstacle-aware kelią,
posūkius, greitį ir nusileidimą. `S3_RTH`, `S_LANDED`, `S_FAIL`, taip pat į
grįžimą eskalavęs `S_OBS`, nėra `eligible_executors()` kandidatai. Perskirstymas
negali atšaukti įsipareigoto grįžimo.

Lloyd ENERGY planavimo kontekstas per `return_energy_for_drone` kviečia tą patį
calculator su `DroneEnergyState.base`; `agl_m` lemia tik neįvykusį kilimą.
`reserve_j` atimamas vieną kartą biudžete. `emergency_frac=None` ir toliau yra
atskiras 20 % avarinis slenkstis, ne dinaminio RTH rezervo pakaitalas.

ZONE_COMPLETE coherent vykdyme publikuojamas per bendrą
`Agent._announce_zone_complete()`. Vieno žingsnio hold leidžia varikliui
pritaikyti reviziją, tačiau `CoherentFlight.step()` pirmiausia tikrina avarinę
bateriją, RTH ir obstacle threat, todėl hold negali užlaikyti būtino grįžimo.

## Saugaus judėjimo darbo prijungimo taškas

Kiti darbai turi jungti faktinius laiku pažymėtus segmentus prie
`CoherentFlight.tick()` (kur realiai keičiama poza ir skaičiuojama energija)
ir/arba `SimulationEngine.run()` po agentų žingsnio. Dabartinis registratorius
`history.record_position(...)` įrašo tik post-tick 2D pozas, o
`ViolationRecorder.observe(...)` stebi pre/post airborne aibę; tai nėra
segmentų tarp žingsnių susikirtimų įrodymas.

Negalimi pažeisti invariantai:

- naujas transit privalo prasidėti tiksliai esamoje pozoje ir heading;
- RTH turi naudoti individualų base bei esamą AGL ir negali būti atšauktas
  perskirstymu;
- saugos perėmimas turi turėti pirmumą prieš repartition hold;
- agento baterija, energijos integralas, rasteris ir foto istorija yra
  monotoniški per reviziją;
- planavimo kandidato atmetimas neturi sukelti dalinio būsenos pakeitimo.

Wakes, S1_TRANSIT/S3_RTH separation išimtys, konfliktų sprendimas, faktinių
3D segmentų registravimas, tarp-tick susikirtimų, greičio ir vertikalaus
tęstinumo tikrinimas nėra REV-02 apimtis. Jie nebuvo išjungti, bet dabartinis
diagnostinis kelias jų dar nepatvirtina.

## Regresiniai įrodymai

- `tests/integration/test_coherent_lloyd_repartition.py`: CVT ir ENERGY
  coherent revizija, fizinės būsenos tęstinumas, failure, netinkami BO,
  kandidato atmetimo atomika ir mission contract eksportas.
- `tests/unit/execution/test_exp09_coherent.py`: likęs kilimas, asmeninis base,
  RTH prognozės/vykdymo energija, vienas rezervas ir cache invalidacija.
- `tests/integration/test_exp09_routes_energy.py`: ankstesnis ConfigError
  lūkestis pakeistas abiejų Lloyd metodų faktine perskirstymo elgsena.
