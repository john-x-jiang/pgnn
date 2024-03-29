# Running guidance

1. Check your cuda version. I've tested `cu116` and it is able to work.
2. Python version >= `3.9` would be recommended.
3. To install the python packages, change the variable `CUDA` in the script `req_torch_geo.sh`, then run it.
4. Check the configuration under `./config/`. Try `demo01.json` for unsupervised SSM. Try `demo02.json` for different time resolution. Try `demo03.json` for mixed SSM.
5. To train the model, run the following command:
   ```bash
   python main.py --config demo --stage 1
   ```
6. To evaluate the model, run the following command:
   ```bash
   python main.py --config demo --stage 2
   ```
7. To make graphs based on the pre-defined geometry, run the following command:
    ```bash
   python main.py --config demo --stage 0
   ```
