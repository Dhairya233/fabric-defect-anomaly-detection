# Voiceover script: main

Timings come from the synthesised voice. Every number is read from `site/data/results.json` and `results/margin_summary.csv` when the video is built.

| Time | Scene | Line |
|---|---|---|
|    0.6s | hook | The same random forest scored 0.23 or 0.68, depending only on how the data was split. |
|    9.6s | problem | This project tests classical computer vision on 245 fabric strips. |
|   15.2s | problem | Strips are cut into overlapping patches. Split by patch, and test texture leaks into training. |
|   21.5s | problem | So I split by whole strip, and held out entire fabrics. |
|   26.3s | methods | Each patch gets ninety texture features. Supervised models learn defects. Anomaly detectors learn only normal fabric. |
|   34.8s | methods | Everything is scored three ways, including a deliberately leaky control. |
|   40.5s | explorer | This is the results explorer. |
|   42.8s | explorer | For the random forest, patch precision is 0.23 split by strip, and 0.68 with leaky patches. A paired difference of plus 0.44, interval 0.27 to 0.53. |
|   57.9s | explorer | On an unseen fabric, nothing clearly beats chance. The best of eight methods scores 0.59, interval 0.44 to 0.72. |
|   68.7s | explorer | Every strip can be inspected. Here are the forest's misses. |
|   72.4s | explorer | Some results look good for the wrong reasons. Brightness alone reaches 0.83 within one fabric. Those strips were captured differently. |
|   82.5s | limits | The limits are real: one small dataset, 7 fabrics, precomputed results, no factory testing. |
|   89.3s | limits | 204 of 245 strips also have blank margins. |
|   95.6s | end | Code, tables and the explorer are all open. |

Total length: 99.3 s.
