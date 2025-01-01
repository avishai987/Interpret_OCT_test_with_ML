The retina is a thin layer of cells at the back of the eyeball. It is the part of the eye that converts light into nerve signals.

Optical Coherence Tomography (OCT) is a non-invasive eye imaging technique using light waves to create cross-sectional images of various parts of the eye, such as the retina and optic nerve. This test allows ophthalmologists to monitor tissues in a way that is not possible with clinical examinations and helps diagnose various eye diseases.

In our project, we received OCT scans and aim to find an algorithm that determines whether there is a retinal disease, and if so, which disease, from three possible diseases.

The diseases are:

CNV - Pathological blood vessel growth in the retina in patients suffering from severe myopia. Significant fluid amounts can be seen with arrows.
DME - Diabetic macular edema. This retinal disease is caused by increased blood flow in the retinal blood vessels, leading to damage to their walls. Fluid thickening can be seen with arrows.
DRUSEN - Protein-like deposits accumulating in the macula (the center of the retina) that may cause loss of visual acuity and other eye problems. The arrows represent several drusen.
NORMAL - Normal retina. Examples:
![image](https://github.com/user-attachments/assets/63b1409a-69a9-4bdb-b381-911c4d12c57e)


The database contains 84,495 images in JPEG format, divided into train, test, and validation sets. Each year, 30 million OCT scans are performed, and the analysis and interpretation of each scan take significant time. Therefore, a tool that can classify diseases accurately is crucial for effective diagnosis and treatment. For example, patients with DME or CNV will need immediate medical intervention, whereas patients with DRUSEN will require monitoring.

Data link: 
https://data.mendeley.com/datasets/rscbjbr9sj/2

Results (in Hebrew): https://github.com/avishai987/Interpret_OCT_test_with_ML/blob/main/Report-%20Hebrew.pdf
