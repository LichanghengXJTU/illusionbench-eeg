"""Pure classical CV stimulus generation for IllusionBench-EEG.

User direction (2026-05-25): no generative AI / no DNN; pre-trained classical
models (dlib HOG+SVM, dlib 68-pt regression-trees landmark, OpenCV Haar
cascades) are OK. Pipeline = dlib landmarks + OpenCV seamlessClone (Poisson)
+ Reinhard LAB color transfer. Source = filtered in-the-wild FFHQ subset.
"""
