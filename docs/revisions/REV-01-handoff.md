# REV-01 perdavimas: pasiekiamas Lloyd zonos įėjimas

## Pakeista sąsaja

`planning.energy_balance.resolve_reachable_zone_entry(ctx, drone,
work_geometry, centroid_xy, fallback_pose)` atskiria matematinį Lloyd centroidą
nuo fizinio įėjimo. Ji grąžina `ResolvedZoneEntry(anchor_pose, ferry_path)`.
`estimate_fast_from_area(...)` priima pasirenkamą `work_geometry`; Lloyd energijos
politika ją perduoda kiekvienoje iteracijoje ir galutiniame `refresh`.

REV-02 turi perduoti tikrą einamąją `DroneEnergyState.pose`. Negalima šios
pozicijos pakeisti bendru `launch_pose` ar konstravimo metu užfiksuotu startu.

## Invariantai

- Lloyd assignment, site update ir CVT fiksuotas taškas tebenaudoja ploto
  centroidą; fizinio anchor parinkimas jų nekeičia.
- Kai obstacle-aware transit įjungtas, kiekviena priskirto polygoninio darbo
  dalis turi visa tilpti tame pačiame routerio `flyable_region` komponente kaip
  dabartinė BO pozicija. Vienos dalies pasiekiamumas nepatvirtina kitų.
- Anchor priklauso priskirtam darbui ir routerio laisvai erdvei. Ferry energija
  skaičiuojama iš to paties validuoto `Path` per `EnergyModel.path_energy`.
- Kandidatai ir MultiPolygon dalys rikiuojami deterministiškai. Tiesioginis
  teisėtas kelias turi pirmenybę; tarp vienodos klasės kandidatų laimi arčiausias
  matematinio centroido, o lygybę išsprendžia kanoninė geometrijos tvarka.
- Jei nė vienas taškas ar visa darbo dalis nepasiekiama, lieka
  `RouteUnavailable` su konkrečia komponento/routerio priežastimi. Nėra nulinių
  sąnaudų, fiktyvaus chord ar bendro išimties slėpimo.
- Tuščia zona lieka dabartinėje BO pozicijoje. RTH rezervo politika nepakeista.
- Kai aplinkos nėra arba `coverage.transit_free_space` išjungtas, išlaikytas
  ankstesnis fast-estimate elgesys.

## Regresiniai įrodymai

- `tests/unit/planning/test_lloyd_reachable_anchor.py`: raw ir clearance
  centroidai, skylė, deterministinis MultiPolygon, blogo kandidato fallback,
  nepasiekiama dalis, gyva pozicija ir ploto apskaita.
- `tests/integration/test_rev01_lloyd_reachable_anchor.py`: tikra M4E, 5 BO,
  seed 42, replication 1 diagnostika pereina ankstesnį raw kliūtyje buvusio
  centroido energijos įvertį ir gauna teisėtą anchor bei baigtinę ferry energiją.

## Kas lieka REV-02 ir vėlesniems darbams

REV-02 sujungia `execution_coherent` su `repartition_enabled`, konkretaus BO base
su RTH energija ir plano revizijų eksportu. REV-01 nekeičia redistribucijos
metodikos. Wakes, separation tranzito/RTH fazėse, faktinių segmentų saugos
registravimas, YAML/CLI eksperimento integracija, metrikos ir 30 kartojimų
patikra taip pat lieka vėlesniems darbams.
