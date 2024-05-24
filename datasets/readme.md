Dataset from  THE ALGONAUTS PROJECT

The 2023 Challenge:
How the Human Brain Makes Sense of Natural Scenes

 (http://algonauts.csail.mit.edu/challenge.html)


Train Data

Images: For each of the 8 subjects there are [9841, 9841, 9082, 8779, 9841, 9082, 9841, 8779] different images (in '.png' format). As an example, the first training image of subject 1 is named 'train-0001_nsd-00013.png'. The first index ('train-0001') orders the images to match the stimulus images dimension of the fMRI train split data. This indexing starts from 1. The second index ('nsd-00013') corresponds to the 73,000 NSD image IDs that you can use to map the image back to the original '.hdf5' NSD image file (which contains all the 73,000 images used in the NSD experiment), and from there to the COCO dataset images for metadata). The 73,000 NSD images IDs in the filename start from 0, so that you can directly use them for indexing the '.hdf5' NSD images in Python. Note that the images used in the NSD experiment (and here in the Algonauts 2023 Challenge) are cropped versions of the original COCO images. Therefore, if you wish to use the COCO image metadata you first need to adapt it to the cropped image coordinates. You can find code to perform this operation here.

fMRI: Along with the train images we share the corresponding fMRI visual responses (as '.npy' files) of both the left hemisphere ('lh_training_fmri.npy') and the right hemisphere ('rh_training_fmri.npy'). The fMRI data is z-scored within each NSD scan session and averaged across image repeats, resulting in 2D arrays with the number of images as rows and as columns a selection of the vertices that showed reliable responses to images during the NSD experiment. The left (LH) and right (RH) hemisphere files consist of, respectively, 19,004 and 20,544 vertices, with the exception of subjects 6 (18,978 LH and 20,220 RH vertices) and 8 (18,981 LH and 20,530 RH vertices) due to missing data.

Test Data

Images: For each of the 8 subjects there are [159, 159, 293, 395, 159, 293, 159, 395] different images (in '.png' format). The file naming scheme is the same as for the train images.

fMRI: The corresponding fMRI visual responses are not released.

Region-of-Interest (ROI) Indices

The visual cortex is divided into multiple areas having different functional properties, referred to as regions-of-interest (ROIs). Along with the fMRI data we provide ROI indices for selecting vertices belonging to specific visual ROIs, that Challenge participants can optionally use at their own discretion (e.g., to build different encoding models for functionally different regions of the visual cortex). However, the Challenge evaluation score is computed over all available vertices, and not over any single ROI. For the ROI definition please see the NSD paper. Note that not all ROIs exist in all subjects. Following is the list of ROIs provided (ROI class file names in parenthesis):

Early retinotopic visual regions (prf-visualrois): V1v, V1d, V2v, V2d, V3v, V3d, hV4.

Body-selective regions (floc-bodies): EBA, FBA-1, FBA-2, mTL-bodies.

Face-selective regions (floc-faces): OFA, FFA-1, FFA-2, mTL-faces, aTL-faces.

Place-selective regions (floc-places): OPA, PPA, RSC.

Word-selective regions (floc-words): OWFA, VWFA-1, VWFA-2, mfs-words, mTL-words.

Anatomical streams (streams): early, midventral, midlateral, midparietal, ventral, lateral, parietal.
 
ROIs surface plots. Visualizations of subject 1 ROIs on surface plots. Different ROIs are represented using different colors. The names of missing ROIs are left in black.

Development Kit

We provide a Colab tutorial in Python where we take you all the way from data input to Challenge submission. In particular, we show you how to:

Load and visualize the fMRI data, its ROIs, and the corresponding image conditions.

Build linearizing encoding models using a pretrained AlexNet architecture, evaluate them, and visualize the resulting prediction accuracy.

Prepare the predicted brain responses to the test images in the right format for submission to the Challenge website.
