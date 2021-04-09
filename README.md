# SEFNAC methods


The source codes, experimental test-beds and, the datasets of our paper entitled 
"Summable and Nonsummable Data-Driven Models for Community Detection in Feature-Rich Networks" by [Soroosh Shalileh](https://www.hse.ru/en/org/persons/316426865) and, [Boris Mirkin](https://www.hse.ru/en/staff/bmirkin), submmited to the journal of Social Network Analysis and Mining (SNAM).


For more information on how to call our algorithm "SEANACs" or "SEFNACn" one can 
refer to the demo jupyter notebooks "demo.ipynb". 

Also these algorithm can be run through the terminal by calling:
        
        python SEANACs.py/SEFNACn.py --Name="name of dataset in data dir" --PreProcessing="z-m" --Run=1 

  Note that the above method for calling our proposed algorithms requires the dataset to in .pickle format as it provided in data directory.  


For generating similar synthetic data sets, One should call "synthetic_data_generator.py" as 
this is demonstrated in Jupyter notebook "generate_synthetic_data.ipynb".

