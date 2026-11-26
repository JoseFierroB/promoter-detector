# Modelos de energía dG — `E1_energy_report`

Resultados de `task3_test` (25k secuencias) para el baseline ProkBERT y las integraciones de energía.
Tabla copiada de `results/E1_energy_report/metrics.tsv` en la WS.

== E: task 3 test split
                    model     n  n_pos      auc       ap    brier  brier_skill
                    task3 24988  12424 0.886398 0.898818 0.136928     0.452271
            energy_fusion 24988  12424 0.879236 0.892890 0.140617     0.437514
        energy_fusion_raw 24988  12424 0.880838 0.894322 0.139883     0.440452
               energy_res 24988  12424 0.878696 0.892299 0.140586     0.437638
   energy_residual_logits 24988  12424 0.879218 0.892631 0.140559     0.437748
    energy_exp_energy_mlp 24988  12424 0.874920 0.888853 0.144112     0.423535
energy_exp_energy_mlp_raw 24988  12424 0.874853 0.889052 0.143274     0.426887
     energy_exp_no_energy 24988  12424 0.886399 0.898820 0.136930     0.452263
       energy_w1w2_direct 24988  12424 0.878655 0.892271 0.140616     0.437519

### Modelos, uno por uno

- **task3**: baseline. ProkBERT-mini preentrenado + pooling ponderado de los autores + `Linear(384, 2)`.
  No usa energía. Es la referencia de la tabla.
- **energy_fusion**: directo  en **w2**. pooled 384, dG 80, `Linear(464, 2)`
- **energy_res**: residual **en w2/logits**. `base_clf(pooled)` (384→2) + `energy_clf(dG)` (80→2);
  la rama de energía empieza en cero. Residual lineal.
- **energy_residual_logits**: residual **en w2/logits** con una MLP de energía: `base_clf(pooled)` +
  `MLP(80→32→2)`. La última capa empieza en cero.
- **energy_exp_energy_mlp**: MLP como clasificador. Después del pooling y dropout:
  `[pooled 384 | dG 80] → Linear(464→384) → GELU → Linear(384→2)`. Energía concatenada antes de la MLP.
- **energy_exp_energy_mlp_raw**: igual que la anterior, pero con los 80 valores dG **raw** (kcal/mol),
  sin estandarizar.
- **energy_exp_no_energy**: control sin energía, con el mismo pipeline y datos. Su resultado es
  prácticamente idéntico a `task3` (0.886399 vs 0.886398), como debe ser.
- **energy_w1w2_direct**: energía **en w1 y w2 a la vez**. En w1 se añade una señal de energía por token; en w2 se concatena el vector dG al pooled (`Linear(464→2)`).
  Es la integración directa en ambas etapas lineales.


- `training/train_improved.py` — baseline task3.

- `training/energy_res.py` — `energy_res` (residual lineal en logits).
- `training/energy_training.py` — residual MLP (`energy_head`).
- `training/energy_two_layers.py` — MLP concatenado.
- `training/energy_w1_w2_energy_direct.py` — energía en w1 y w2.
- `evaluation/score_experiments.py` — genera `results/scores`.
- `evaluation/compare_new_copy.py` — genera `results/E1_energy_report` y E2.
- `evaluation/plot_curves_copy.py` — genera `results/training_curves`.

### Lectura rápida

- `task3` es el mejor (AUC 0.8864).
- `energy_exp_no_energy` reproduce el baseline (0.8864), lo que confirma que el pipeline de control es correcto.
- `energy_res` y `energy_residual_logits` quedan ~0.879.
- Las MLP con energía quedan ~0.875; la versión raw no mejora.
- `energy_w1w2_direct` queda ~0.8787.
