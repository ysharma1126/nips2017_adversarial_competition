NIPS 2017 Competition on Adversarial Attacks and Defenses
=====================================

Submissions to the non-targeted attack, targeted attack, and defense tracks for the NIPS Competition

Non-Targeted Attack
-------------------------------------
Directory: pgd/

Implemented a weighted ensembled version of the PGD attack introduced by Madry et al. 

Weighted Ensemble: (0.35,Inception v3), (0.25,Ensemble Adversarially trained Inception ResNet v2), (0.2,Adversarially trained Inception v3), (0.1,Inception ResNet v2), (0.1,ResNet V2-152)

Final Placement: 14/91

Targeted Attack
-------------------------------------
Directory: iter_target/

Implemented a ensembled version of the iterative target class attack introduced by Kurakin et al.

Ensemble: Inception v3, Ensemble Adversarially trained Inception ResNet v2, Adversarially trained Inception v3

Instead of ensembling by each iteration averaging the losses (like what was done for non-targeted), generated adversarial examples individually on each model and averaged the resulting solutions. 

Final Placement: 6/65

Defense
-------------------------------------
Directory: ensemble_defense/

Performed JPEG-Compression with quality=25 on the adversarial images, fed images to the following weighted ensemble. The mode of the predictions was taken. 

Weighted Ensemble: (0.3,Inception ResNet v2), (0.3,Ensemble Adversarially trained Inception ResNet v2), (0.25,ResNet V2-152), (0.15,Adversarially trained Inception v3)

JPEG-Compression quality chosen in order to trade-off reducing the power of strong adversarial examples, while not harming the predictions on clean/weak images. 

Final Placement: 11/107

Experiment
-------------------------------------

Download DEV dataset

```
python dataset/download_images.py --input_file=dataset/dev_dataset.csv --output_dir=dataset/images/
```

Download model_checkpoints needed for the non-targeted attack, targeted attack, and defense by navigating to the model_ckpts folder in each directory, and executing 'download_all.sh'

Generate adversarial images, and evaluate results on a set of undefended models with simple image processing defenses available by executing the commands provided in 'commands.txt'. Docker required, Nvidia-docker if GPUs would like to be used. 
