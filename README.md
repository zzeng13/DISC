
<!-- PROJECT LOGO -->
<br />
<p align="center">


  <h3 align="center">IDentifier of Idiomatic Expressions via Semantic Compatibility (DISC)</h3>

  <p align="center">
   An Idiomatic identifier that detects the presence and span of idiomatic expression in a given sentence. 
    <br />




<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
     <ul>
        <li><a href="#configuration">Configuration</a></li>
        <li><a href="#demo">Demo</a></li>
        <li><a href="#data-processing">Data Processing</a></li>
        <li><a href="#training-testing">Training and Testing</a></li>
      </ul>
     </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project is a supervised idiomatic expression identification method. Given a sentence that contains a potentially idiomatic expression (PIE), the model identifies the span of the PIE if it is indeed used in an idiomatic sense, otherwise, the model does not identify the PIE.  The identification is done via checking the smemantic compatibility. More details will be updated here (Detail description, figures, etc.).

The paper will appear in TACL. 





### Built With

This model is heavily relying the resources/libraries list as following: 
* [PyTorch](https://pytorch.org/)
* [Huggingface's Transformer](https://huggingface.co/)
* [NLTK](https://www.nltk.org/)



<!-- GETTING STARTED -->
## Getting Started
The implementation here includes processed data created for [MAGPIE](https://aclanthology.org/2020.lrec-1.35.pdf) random-split dataset. The model checkpoint that trained with MAGPIE random-split  is also provided. 

### Prerequisites

All the dependencies for this project is listed in `requirements.txt`. You can install them via a standard command: 
```
pip install -r requirements.txt
```
It is highly recommanded to start a conda environment with PyTorch properly installed based on your hardward before install the other requirements. 

### Checkpoint

To run the model with a pre-trained checkpoint, please first create a `./checkpoints` folder at root. Then, please download the checkpoint from Google Drive via this [Link](https://drive.google.com/file/d/1pGX1F03FYWymXcZ0hjJ7kYi3fySW5n54/view?usp=sharing). Please put the checkpoint in the  `./checkpoints` folder.


<!-- USAGE EXAMPLES -->
## Usage
### Configuration 
Before running the demo or experiments (training or testing), please see the `config.py` which sets the configuration of the model. Some parameters there, such as `MODE` needs to be set appropriately for the model to run correctly. Please see comments for more details. 

### Demo
To start, please go through the examples provided in `demo.ipynb`. In there, we process a given input sentence into the model input data and then run model inference to extract the idiomatic expression (if present) from the input sentence (visualized).

### Data processing
To process a dataset (such as MAGPIE) for model training and testing, please refer to `./data_processing/MAGPIE/read_comp_data_processing.ipynb`. It takes a dataset with sententences and their PIE lcoations as input and generate all the necessary files for model training and inference. 

### Training and Testing 
For training and testing, please refer to `train.py` and `test.py`. Note that `test.py` is used to produce evaluation scores as shown in the paper. `inference.py` is used to produce prediction for sentences.
 




<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Ziheng Zeng - zzeng13@illinoie.edu

Project Link: [https://github.com/your_username/repo_name](https://github.com/zzeng13/DISC)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements


### [TODO]: 
Add the following in README: 
* Method detail descrption
* Method figure 
* Demo walkthrough 
* Data processing tips and instructions
Add `requirements.txt`







