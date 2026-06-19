# 平行運算總覽 (Parallelism Overview)

本專案有**兩個層級**的平行運算,刻意分得很清楚:

1. **OpenMP** — 求解器內部的共享記憶體平行(主力,預設啟用)
2. **MPI** — 跨案例的粗粒度任務平行(可選,預設關閉)

---

## 1. OpenMP — 求解器內部共享記憶體平行(主力)

真正在數值熱點迴圈上的平行化,全部集中在 [`src/mhd_runner.cpp`](../src/mhd_runner.cpp),
由 CMake 強制啟用:

- [`CMakeLists.txt:8`](../CMakeLists.txt#L8) — `find_package(OpenMP REQUIRED)`
- 連結到 target:[`CMakeLists.txt:80`](../CMakeLists.txt#L80)、[`CMakeLists.txt:121`](../CMakeLists.txt#L121)

共 5 個 `#pragma omp parallel for`:

| 位置 | 平行的工作 | 形式 |
|------|-----------|------|
| [`mhd_runner.cpp:1264`](../src/mhd_runner.cpp#L1264) | `max_signal_speed_2d`:逐格算最大訊號速度供 CFL 用 | `reduction(max:smax)` 歸約 |
| [`mhd_runner.cpp:1322`](../src/mhd_runner.cpp#L1322) | 逐格把守恆量轉成原始變數 `state_to_prim` | `schedule(static)` |
| [`mhd_runner.cpp:1332`](../src/mhd_runner.cpp#L1332) | MUSCL/PLM 的限制斜率計算 | `schedule(static)` |
| [`mhd_runner.cpp:1358`](../src/mhd_runner.cpp#L1358) | X 方向逐面 Riemann 解(HLLD / LLF flux) | `schedule(static)` |
| [`mhd_runner.cpp:1380`](../src/mhd_runner.cpp#L1380) | Y 方向逐面 Riemann 解 | `schedule(static)` |

後三個位於核心函式 `compute_rhs_blended_2d`(計算 RHS = −div F),也就是
**每個 RK 子步都會呼叫的最貴部分**。

平行安全性(程式碼註解 [`mhd_runner.cpp:1351-1357`](../src/mhd_runner.cpp#L1351-L1357)):
每個面只寫自己唯一的 `flux_x[face]`,而 `compute_flux` / `compute_llf_flux` 是
pure function,因此迭代之間無相依、可安全並行。

### 為什麼平行化在 runner,而不是 HLLD solver?

常見疑問:HLLD / LLF 通量明明是 solver 的工作,為什麼 `#pragma omp` 出現在 runner?
答案是**責任分離** —— 平行化的對象是「迴圈」,不是「一次計算」。

- **Solver([`src/hlld_solver.cpp`](../src/hlld_solver.cpp))只處理「一個面」。**
  `MHD::compute_flux`([`hlld_solver.cpp:424`](../src/hlld_solver.cpp#L424))吃一組
  左右狀態 `(W_L, W_R)`,吐一個通量。它不知道網格、不知道有幾個面、也不知道平行的存在
  —— 是一個 **pure function**。
- **Runner([`src/mhd_runner.cpp`](../src/mhd_runner.cpp))負責「走訪所有面」。**
  `nx × ny` 個面、每個面各呼叫一次 solver,這個數量龐大且彼此獨立的迴圈只存在於 runner,
  也只有它適合套 `parallel for`。

> Solver 回答「**這一個面的通量是多少**」;runner 回答「**怎麼把這個 solver
> 跑遍整張網格**」。平行化是後者的責任。

正因為 solver 保持 pure(不碰共享狀態),runner 才能放心把對它的數萬次呼叫灑到多執行緒;
而且同一個 solver 還被 X 掃描、Y 掃描、不同清理方法、測試重用,把平行策略留在 runner
就不會把 solver 綁死在某一種平行模型上。

---

## 2. MPI — 跨案例的粗粒度任務平行(可選)

位於 [`src/mhd_sweep_mpi.cpp`](../src/mhd_sweep_mpi.cpp),**預設關閉**
(`ENABLE_MPI` 預設 OFF:[`CMakeLists.txt:11`](../CMakeLists.txt#L11)、
[`CMakeLists.txt:148`](../CMakeLists.txt#L148))。

### 用白話講:它在做什麼

想像你要比較好幾種清理方法(no cleaning、hyperbolic GLM、mixed GLM …),
每一種都要跑一次完整模擬。**這些模擬彼此完全無關**,誰先誰後、誰跟誰都不影響結果。

最笨的做法是一個接一個跑(跑完 A 才跑 B)。MPI 在這裡做的事很單純:
**「你有幾顆 CPU,我就同時開幾個,把這些模擬分給它們各跑各的。」**

打個比方 —— 有 8 份考卷要改、4 位老師:

- 老師 1 改第 1、5 份
- 老師 2 改第 2、6 份
- 老師 3 改第 3、7 份
- 老師 4 改第 4、8 份

每位老師改自己分到的,**改的過程完全不用跟別人講話**,改完各自交一疊成績單,
最後再有人把 4 疊併成一張總表。MPI 版做的就是這件事,「老師」在 MPI 術語裡叫 **rank**。

對應到程式碼:

- 程式啟動時問「總共幾個 rank、我是第幾號」
  ([`mhd_sweep_mpi.cpp:349-354`](../src/mhd_sweep_mpi.cpp#L349-L354))
- 用一行 `j % size == rank` 決定「第 j 個模擬歸不歸我跑」—— 這就是上面
  「第 1、5 份歸老師 1」的分配規則
  ([`mhd_sweep_mpi.cpp:371-380`](../src/mhd_sweep_mpi.cpp#L371-L380))
- 每個 rank 把自己的結果寫成獨立的 CSV(`summary_rank_0000.csv`、
  `summary_rank_0001.csv` …)
- 全部跑完後,用 Python 腳本
  [`merge_mpi_sweep_summaries.py`](../scripts/run/merge_mpi_sweep_summaries.py)
  把這些 CSV 併成一張總表

### 重點:這不是「把一個模擬切開來算」

這是最容易誤會的地方,程式自己也在說明文字裡特別強調
([`mhd_sweep_mpi.cpp:89-90`](../src/mhd_sweep_mpi.cpp#L89-L90)、
[`README.md:402-404`](../README.md#L402-L404)):

- 它**不是** domain decomposition(區域分解)。
  區域分解是「把一張大網格切成好幾塊、每個 rank 算一塊,過程中還要不斷交換邊界資料」——
  那種才需要 rank 之間頻繁通訊(halo exchange)。**這裡完全沒有這回事。**
- 這裡每個 rank 跑的是**一個完整、獨立的模擬**,中途互不通訊。
  整支程式裡 rank 之間只「講兩句話」:
  開頭一次 `MPI_Barrier`(大家對齊起跑線,
  [`mhd_sweep_mpi.cpp:368`](../src/mhd_sweep_mpi.cpp#L368))、
  結尾一次 `MPI_Allreduce`(彙整「有沒有人出錯」,
  [`mhd_sweep_mpi.cpp:392`](../src/mhd_sweep_mpi.cpp#L392))。

換句話說,每個 rank 內部跑的仍然是前面那個 OpenMP 版求解器。
所以兩層其實可以疊起來:**MPI 同時跑多個模擬,每個模擬內部再用 OpenMP 多執行緒加速。**

### 建置與執行

```bash
cmake -S . -B build-mpi -DENABLE_MPI=ON
cmake --build build-mpi -j

mpirun -np 4 ./build-mpi/mhd_sweep_mpi \
  --problem divergence_advection --prefix mpi_da
```

---

## 3. MPI — 求解器域分解(真正的擴展性平行,可選)

位於 [`src/mhd_runner_mpi.cpp`](../src/mhd_runner_mpi.cpp) +
[`src/mpi_domain.cpp`](../src/mpi_domain.cpp),同樣由 `ENABLE_MPI` 閘控、預設關閉。
**與第 2 節徹底不同**:第 2 節是「每個 rank 跑一個完整模擬」;這裡是
**把一個模擬切開、每個 rank 算一塊**,過程中不斷交換邊界——這才是 domain decomposition。

### 白話講:它在做什麼

把 `nx_g × ny_g` 的大網格用 `MPI_Cart_create` 切成 `px × py` 塊,每個 rank 拿一塊
`nx_loc × ny_loc` 的子域,外圈再包 `ng = 2` 層 **ghost cell**(因為 PLM 重構在 cell
`i` 需要 `i-1,i,i+1` 的斜率 → 需要 `i-2..i+2`)。每個 RK 子步、每個清理子步之前,
先跟上下左右鄰居用 `MPI_Sendrecv` **交換 halo**(把鄰居最外圈的內部 cell 複製到自己的
ghost),核心 kernel 就能照常在「padded 本地陣列」上跑,內部 cell 拿到的 stencil 輸入
跟單一全域網格完全一樣;最外圈 ghost 的環繞污染因為被丟棄,不影響結果。

### 設計重點:核心 kernel 一行都沒改

平行化全部疊在 runner / 一個新的 `mpi_domain` 層,**HLLD/GLM/EGLM kernel 本體零修改**:

- **Halo 交換**:`exchange_halos`([`src/mpi_domain.cpp`](../src/mpi_domain.cpp))
  兩階段(先垂直、再水平含 ghost 行)以填滿對角 ghost;週期性由
  `MPI_Cart_create(periods={1,1})` 自動處理。
- **全域歸約**:CFL 的 `max`、min pressure/density 的 `min`、失敗旗標的 `OR` 走
  `MPI_Allreduce`,所以每個 rank 的 `dt` 與「停 / 重試 / 接受」分支完全一致(否則會在
  集體 halo 交換處死鎖)。這些都是冪等運算,在 ghost 陣列上做也安全。
- **I/O**:所有快照 / 診斷 / summary CSV 由 rank 0 **gather** 後以原格式寫出,
  下游 Python 繪圖與 check 腳本完全不用改。
- **正確性**:回饋進演化的全域量只有 `max`(dt)與 `min`(positivity),兩者順序無關、
  FP 精確,所以非投影方法的演化在多 rank 下應與序列版一致到 round-off。

### 建置與執行

```bash
cmake -S . -B build-mpi -DENABLE_MPI=ON
cmake --build build-mpi -j

mpirun -np 4 ./build-mpi/mhd_runner_mpi \
  --nx 256 --ny 256 --tfinal 0.5 orszag_tang mixed_glm
```

`nx_g/ny_g` 必須能被 `MPI_Dims_create` 選出的程序格維度整除;`-np 1` 等同序列結果。

### elliptic_projection 的分散式 Poisson 解

多 rank 橢圓投影使用 matrix-free conjugate gradient 解週期 Poisson 方程。
每個 rank 只配置含 ghost cells 的局部 scalar arrays；每次 Laplacian 操作前交換
`phi` halo，dot product 與 residual norm 則使用 `MPI_Allreduce`。右手邊先減去
全域平均值以移除週期 Laplacian 的 nullspace，解完成後也將 `phi` 設為零平均。
`-np 1` 保留原本的序列 SOR 路徑。

---

## 4. 沒有用到平行的地方(值得知道)

- **時間推進迴圈本身是串行的** — RK 子步之間有相依;平行只發生在每個子步
  「內部」的逐格 / 逐面迴圈(OpenMP 在執行緒間、域分解在 rank 間)。
- **所有 Python 腳本都是串行的** — `scripts/run/run_performance_scaling.py`
  等是逐一 `subprocess` 呼叫;README 也明說效能 scaling 量的是
  "serial wall-clock"([`README.md:62`](../README.md#L62)、
  [`README.md:403-404`](../README.md#L403-L404))。
  整個 `scripts/` 下沒有 `multiprocessing` / `concurrent.futures` / `mpi4py`。
- 沒有 GPU / CUDA、沒有 `std::thread`。

---

## 一句話總結

核心 2.5D MHD 求解器靠 **OpenMP** 在每個 RK 子步內把逐面 HLLD 通量、
原始變數轉換、斜率、CFL 歸約平行化(共享記憶體,主力、預設啟用)。
另有兩個預設關閉的 **MPI** 層:`mhd_sweep_mpi` 做「不同模擬案例分散到不同 rank」的
粗粒度任務平行;`mhd_runner_mpi` 則是**真正的 2D 域分解**——把單一模擬切成子域、
每個 rank 算一塊、每子步交換 halo、全域量用 `MPI_Allreduce`、rank 0 gather 輸出,
讓單一大模擬可隨 rank 擴展(elliptic_projection 除外)。三層可疊加:
MPI 切網格 × 每個 rank 內 OpenMP 多執行緒。
