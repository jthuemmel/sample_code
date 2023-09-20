## Sample implementations of various models I worked with during my PhD. 
Training and analsyis code is (currently) not included. All model code was written and tested by myself over the last two years, with inspiration from the official implementations and https://github.com/lucidrains. 
Most recent work was done on SwinLSTM, PerceiverIO and MAE for ENSO ocean data.

### Contents:
- data
  - enso: dataset and preprocessing (courtesy Jakob Schlör) for ocean data from CMIP6.
  - weatherbench: dataset and scores (courtesy Stephan Rasp) for WeatherBench 1.
  - 2dwaves: dataset and data generators (courtesy Matthias Karlbauer) for 2d wave equation.
- losses:
  - latMSE for weighing losses according to their latitude for global earth data.
  - loss_fn contains three probabilistic loss functions: NormalCRPS (Gneiting 2005), BetaNLL (Seitzer et al 2022), StatisticalLoss (Lessig et al 2023).
- models
  - attention contains implementations based on AFNO (Guibas et al 2021), Vision Transformer (Dosovitskiy et al 2021), PerceiverIO (Jaegle et al 2021) and Masked Auto-Encoder (He et al 2021).
  - conv_rnn contains versions of ConvLSTM (Shi et al 2015), ConvGRU (Ballas et al 2016), Distana (Karlbauer et al 2019) as well as experimental versions of these models.
  - gnn contains a barebones implementation of a GNN as per Keisler 2022.
  - swin_lstm contains an updated version of ConvLSTM for larger receptive fields and with learnable conditioning (as in Perez et al 2017).
- notebooks
  - mnist_gnn is a showcase tutorial of a GNN on MNIST, made for a student.
  - enso_cnn_classifier is a sample implementation of a CNN (based on Liu et al 2022) for use in el nino event classification, made for a student.
