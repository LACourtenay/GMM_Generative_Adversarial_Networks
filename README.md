# GMM_Generative_Adversarial_Networks
Generative Adversarial Networks for Geometric Morphometric Data Augmentation



-----------------------------------------------------------------------------------------------------------------

<i>
Author

Lloyd A. Courtenay

Email

ladc1995@gmail.com

ORCID

https://orcid.org/0000-0002-4810-2001

Current Afiliations:
TIDOP Group of the Department of Cartographic and Land Engineering (TIDOP)

</i>

---------------------------------------------------------------------------------------------------

This code has been designed for the open-access free python software.

--------------------------------------------------------

Please cite these codes as:

 <b> Courtenay (2020) Generative Adversarial Networks for Geometric Morphometric Data Augmentation. https://github.com/LACourtenay/GMM_Generative_Adversarial_Networks
</b>

--------------------------------------------------------

This code was designed and prepared for the study by:
<b> Courtenay (Submitted to Evolutionary Anthropology) Geometric Morphometric Data Augmentation using Generative Computational Learning Algorithms </b>

------------------------------------------------------------
Comments, questions, doubts, suggestions and corrections can all be directed to L. A. Courtenay at the email provided above.
Science should be a collaborative effort. Archaeology is no exception to this. If any fellow data
scientist wishes to improve, contribute, update, report issues or help with this work, please do not hesitate to contact the corresponding author.

--------------------------------------------------------
<b> Instructions on use: </b>
In order to use the code the analyst should prepare a new folder as a directory for data science applications. This folder should contain a data.txt file containing
the dataset. All values should be seperated by commas. In the case of simple GAN applications no labels should be included within this file, for Conditional GAN (CGAN)
applications the first value within this file should be the label. For GAN applications the analyst should first scale their data via the Scaling_Data.py file. This will produce a new file within the directory containging the scaled data. The analyst should then procede to augment with either the LSGAN, WGAN or WGAN_GP files. Each of these files will fill the directory with checkpointed generator .h5 files, plots of the augmented datasets at each step of the training process, as well as a final rning curve plot. The generator files can then be used to augment datasets via Using_Trained_Generator_to_Augment_Datasets.py, producing a large .csv file containing all of the augmented data at different points of the training process. This .csv file can then be used to evaluate GAN performance. Once the optimal GAN has been defined the final step within the Using_Trained_Generator_to_Augment_Datasets.py file can be used to produce the final augmented dataset. For CGAN applications no prior scaling is needed, this function is included within the CGAN_WGAN_GP.py file. For optimal results and to avoid errors, I strongly recomend loading these files in a software or text editor with debugging applications, such as Jupyter Notebook. If the analyst includes any other sub-directories within their main directory, or changes the name or format of any files, then they will be required to modify the code provided to avoid errors.

