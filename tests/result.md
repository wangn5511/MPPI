# MVP 五个子任务测试结果

## 1) 端侧闭环执行时序（按频率消费 action chunk）

```bash
== Playback Summary ==
steps: 200
infer_ms mean: 44.94904888328165
infer_ms p50 : 44.482744531705976
infer_ms p95 : 46.67431563138961
infer_ms max : 67.27478886023164
unique policies: 65
   mppi_joint+curobo+ess0.830+tab1+cub3+sph41
   mppi_joint+curobo+ess0.831+tab1+cub3+sph41
   mppi_joint+curobo+ess0.843+tab1+cub3+sph41
   mppi_joint+curobo+ess0.844+tab1+cub3+sph41
   mppi_joint+curobo+ess0.848+tab1+cub3+sph41
   mppi_joint+curobo+ess0.849+tab1+cub3+sph41
   mppi_joint+curobo+ess0.850+tab1+cub3+sph41
   mppi_joint+curobo+ess0.851+tab1+cub3+sph41
   mppi_joint+curobo+ess0.858+tab1+cub3+sph41
   mppi_joint+curobo+ess0.859+tab1+cub3+sph41
```

## 2) cuRobo 自碰 self-collision 的启用与有效性

```bash
== Playback Summary ==
steps: 50
infer_ms mean: 6.495910361409187
infer_ms p50 : 6.320232525467873
infer_ms p95 : 7.446401030756532
infer_ms max : 7.746322080492973
unique policies: 20
   mppi_joint+curobo+ess0.946+tab0+cub0+sph0
   mppi_joint+curobo+ess0.950+tab0+cub0+sph0
   mppi_joint+curobo+ess0.951+tab0+cub0+sph0
   mppi_joint+curobo+ess0.952+tab0+cub0+sph0
   mppi_joint+curobo+ess0.955+tab0+cub0+sph0
   mppi_joint+curobo+ess0.956+tab0+cub0+sph0
   mppi_joint+curobo+ess0.957+tab0+cub0+sph0
   mppi_joint+curobo+ess0.958+tab0+cub0+sph0
   mppi_joint+curobo+ess0.959+tab0+cub0+sph0
   mppi_joint+curobo+ess0.960+tab0+cub0+sph0
```
## 3) 更新策略必要性验证（A/B 场景交替导致的闪烁）

```bash
== Playback Summary ==
steps: 200
infer_ms mean: 93.65502858301625
infer_ms p50 : 93.63348898477852
infer_ms p95 : 95.0465643312782
infer_ms max : 96.81511297821999
unique policies: 88
   mppi_joint+curobo+ess0.735+tab1+cub4+sph41
   mppi_joint+curobo+ess0.736+tab1+cub4+sph41
   mppi_joint+curobo+ess0.751+tab1+cub4+sph41
   mppi_joint+curobo+ess0.762+tab1+cub4+sph41
   mppi_joint+curobo+ess0.763+tab1+cub4+sph41
   mppi_joint+curobo+ess0.766+tab1+cub4+sph41
   mppi_joint+curobo+ess0.769+tab1+cub4+sph41
   mppi_joint+curobo+ess0.770+tab1+cub4+sph41
   mppi_joint+curobo+ess0.771+tab1+cub4+sph41
   mppi_joint+curobo+ess0.772+tab1+cub4+sph41
```

## 5) time budget 压力测试（为降级策略提供依据）

```bash
== Playback Summary ==
steps: 200
infer_ms mean: 23.052096699830145
infer_ms p50 : 22.483322769403458
infer_ms p95 : 24.24152479507029
infer_ms max : 66.26260187476873
unique policies: 51
   mppi_joint+curobo+ess0.893+tab1+cub7+sph41
   mppi_joint+curobo+ess0.896+tab1+cub7+sph41
   mppi_joint+curobo+ess0.901+tab1+cub7+sph41
   mppi_joint+curobo+ess0.903+tab1+cub7+sph41
   mppi_joint+curobo+ess0.905+tab1+cub7+sph41
   mppi_joint+curobo+ess0.906+tab1+cub7+sph41
   mppi_joint+curobo+ess0.908+tab1+cub7+sph41
   mppi_joint+curobo+ess0.909+tab1+cub7+sph41
   mppi_joint+curobo+ess0.910+tab1+cub7+sph41
   mppi_joint+curobo+ess0.911+tab1+cub7+sph41
```