# gsoc2025ML4Sci

The purpose of EXXA is to use machine learning to analyze exoplanets, their systems, and their formation. 

Astronomical data often takes the form of images, so expertise in computer vision techniques is often essential for understanding and working with observations. Telescopes, such as ALMA, Kepler and NASA, produce publicly available data that can be used (with proper citations) by amateur and professional researchers alike. However, research often supplements this data with synthetic data from simulations.

# Clustering Results curve - General Test

Protoplanetary disks are the sites of planet formation. The most recent generation of telescopes is able to resolve these objects in sufficient detail for the rigorous study of their properties, leading to a dramatic and rapid advancement in planet formation theory. Using synthetic observations that mimic data obtained from these observatories has allowed researchers to understand how specific conditions will manifest themselves in observations.


<img width="633" alt="Clustering Results curve" src="https://github.com/user-attachments/assets/c62f64a3-729f-4bdd-9b71-dc0dfb8df00a" />

# Observations from GMM Clustering and Data Analysis:

After applying Gaussian Mixture Modeling (GMM), five distinct clusters were observed through a 2D plot.

The clusters are visualized within a range of:

x-axis (t-SNE 1): -10 to 10
y-axis (t-SNE 2): -10 to 10

The density or accumulation of these clusters indicates an increased probability of the presence of a planet in those regions of the feature space. Notably:

Region 1 (Violet): Exhibits the highest density of clustering, indicating a greater likelihood of planet presence.
Region 3 (Light Green): Shows the lowest cluster density, suggesting a lower probability of detecting a planet in that area.

The provided planetary data is incomplete, as some FITS files contain null or missing values at the 0th index of certain dimensions. This may affect the accuracy of the model and needs to be handled during preprocessing or data cleaning stages.

# ROC Curve - Sequential Test

Astronomers have several methods at our disposal to detect exoplanets. Historically, many of the first exoplanets to be discovered were found via the radial velocity method. Using super-precise and ultra-well calibrated instruments, astronomers searched for the tiny wobble in the star's speed caused by the planet's orbit. Other methods at exoplanet astronomers' disposals include detecting gravitational lensing due to a planet (called the microlensing method), searching for the wobble in the star's position on the sky (called the astrometric method), and separating the light of the star from the planet and actually taking images (called the direct imaging method). One of the most successful methods to detect exoplanets is using light curves. Several thousand planets have been discovered this way. The basic idea is that exoplanets crossing in front of their host stars will obscure part of the star, which decreases the amount of light that we see from that star. By carefully measuring the brightness over time, planets can be identified by the periodic dimming. The extent of the dimming depends on the specific parameters of the stellar system.

<img width="527" alt="ROC curve" src="https://github.com/user-attachments/assets/a7ce6c41-38e7-4ab8-a23a-d8232531978f" />

Model Evaluation and Observations:

A sample evaluation was conducted using n = 10 data points.

The ROC curve plotted for this evaluation forms an L-shaped curve, indicating a strong classifier.

X-axis: False Positive Rate (range: 0.0 to 1.0)
Y-axis: True Positive Rate (range: 0.0 to 1.0)

The model demonstrates excellent performance with an AUC (Area Under Curve) of 0.99, highlighting its high capability to distinguish between positive and negative classes.

Light Curve Analysis:

A separate graph was plotted with brightness vs. time, where a noticeable dip in brightness was observed.

This dip indicates a potential transit event, which can be interpreted as the presence of a planet passing in front of a star—an essential cue in exoplanet detection.

Summary:

The L-shaped ROC curve signifies that the model maintains a low false positive rate while achieving a high true positive rate.

This confirms that the model is reliable in detecting potential planets with minimal error.
