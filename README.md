


# MLF Project

In order to execute the main script is sufficient to run "runCC.m".  
  
Interactions with the program are done through MATLAB console.  
  
The script will ask, one time for every run, for the path of the dataset 
to clusterize.  
  
In the main loop one can choose among different techniques of 
Feature-Extraction and Clustering by submitting the proper index (integer) 
to the console.  
All the Clustering techniques need a parameter in order to work, in particular:  
 - BSAS: max number of clusters (integer);  
 - Mean Shift: window size (double);  
 - Expectation Maximization: number of cluster (integer).  
  
It's also possible to save the result of each computation.

--> If a particular result has been already computed ans saved the <-- 
--> script will automatically load it wihout computing it again!   <--
--> To compute the specific result again is necessary to delete it <--
--> from './save' directory.                                       <--

The loop end with the visualization of the mean image of each cluster and 
asking to repeat again the computation. 




