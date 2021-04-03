# reid-tiny-production
ReID model to deploy for production mode

Information about this project:
- It is reference from "Tiny Person ReID Baseline", https://github.com/lulujianjie/person-reid-tiny-baseline
- It does not include train and test feature, trained model and database
- Model was trained with Market1501 (still fine tuning in process), and inference process was tested with same dataset
- Gallery features of test images were stored in database (sqlite) which is excluded

Inference Process Flow:
![image](https://user-images.githubusercontent.com/39640791/113481779-23385400-94ce-11eb-8b09-11b14f64203c.png)
