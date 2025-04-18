# SERS nose arrays based on a signal differentiation approach for TNT gas detection

This is the artifact of paper SERS nose arrays based on a signal differentiation approach for TNT gas detection.

## Dependency
* Python 3.7
* scikit-learn 0.22.1
* matplotlib 3.2.1
* pandas 1.0.3
* numpy 1.18.3

## Content
The file structure of the artifact is as follows:
* **src:**
  * **1_preprocessing_cs.py:** source code to preprocess the original sers data.
  * **2_mergejson_cs.py:** source code to merge the preprocessed data to json file.
  * **3_ML_10fold_cs.py:** source code to machine learning models and Fig.5.c, Fig.5.d and Fig.5.e.
  * **4_plot_fig5b.py:** source code to Fig.5.b.

* **data:**
  * **xxduBYS:** data for 2,4-DNPA at xx temperature.
  * **xxduTNT:** data for TNT at xx temperature.
  * **25duTNT_original:** original SERS data example for TNT at 25 degrees Celsius.
  * **compare/{tem1}{object1}{tem2}{object2}:** comparison json data for object1 and object2 at two same or different temperature.

## To run the code

* To run this artifact, you need to install all dependencies first.
* Preprocess the original data to get the json files in **xxduBYS** and **xxduTNT**:
```
python 1_preprocessing.py
```
* Merge the data to get the comparison json files in **compare/{tem1}{object1}{tem2}{object2}:**:
```
python 2_mergejson.py
```
* Run the machine learning models and draw figures of the Fig.5.c, Fig.5.d and Fig.5.e:
```
python 3_ML_10fold.py
```
* Draw the Fig.5.b:
```
python 4_plot_fig5b.py
```

## Citation
If you find this code to be useful for your research, please consider citing:

```
comming soon...
```



