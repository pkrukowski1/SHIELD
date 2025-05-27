# SHIELD (Secure Hypernetworks for Incremental Expansion Learning Defense)

**Patryk Krukowski, Łukasz Gorczyca, Piotr Helm, Kamil Książek, Przemysław Spurek** @ *GMUM JU*

🚀 *Let's forget about catastrophic forgetting!* 🚀

## Commands
**Setup**
```
conda env create -f environment.yml
cp example.env .env
edit .env
```

**Launching Experiments**
```
conda activate shield
WANDB_MODE={offline/online} HYDRA_FULL_ERROR={0/1} python src/main.py --config-name config 
```

## Acknowledgements
- Project Structure based on [template](https://github.com/sobieskibj/templates/tree/master) by Bartłomiej Sobieski
- The original implementation of the project was based on [HyperMask](https://github.com/gmum/HyperMask)
