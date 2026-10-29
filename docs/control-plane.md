# Control plane

1. `setup.sh` verifies the target snapshot.
2. `setup.sh` builds the TensorFold worker.
3. `tools/resident-up.sh` starts one serial or DFlash resident.
4. The benchmarker starts a standard-input worker for each phase.
5. `tools/ranked-box-preflight.sh` checks pins before a ranked run.
6. `tools/record-goldens.sh` records serial and `.dflash1` files on the ranked Mac.
