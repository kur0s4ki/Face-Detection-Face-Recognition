# Face-Detection-Face-Recognition

## First Use Case : Access Control / ID Affectation / Distance Estimation

* AUTO ENROLLMENT.  **(DEFAULT : Enabled)**
  * *Enabled* : The user is automatically added to the Database , if he is within the detection threshold.
  * *Disabled* : The user is labeled as UNKNOWN whithin the detection threshold , Pressing **S** , adds the user to the Database.
      
* ID AFFECTATION.   **(DEFAULT METHOD : Counter)**
  * *Counter* : Affect a an ID to a user based on a counter.
  * *UUID_Generator* : Affect an ID to a user based on uuid.
      
* DISTANCE THRESHOLD.   **(DEFAULT VALUE : 50)**
  * *50* : Distance at which detection process starts. the unit is CM.
* FACE DETECTION MODEL.   **(DEFAULT VALUE : hog)**
  * *hog* : Better suited for normal machines. **(FAST)**
  * *cnn* : Better preecision , but requires heavy computations. **(USE ONLY WHITH A GPU)**
      
* NUMBER OF JIITERS.   **(DEFAULT VALUE : 1)**
  * *1* : Default value , how many times the each detected face will be distorted , and manipulated , **HIGHER NUMBER == LOW FPS**
      
* DISTANCE ESTIMATION METHOD.   **(DEFAULT VALUE : Algebra)**
  * *Algebra* : Uses geometric formulas about eyes positions , to estimate the distance. **(Better Estimation)**
  * *Estimation* : Uses special formula to estimate distance.
  * *Perimeter* : Uses the opencv box perimeter to estimate distance.
  
# IMPORTANT 

## DummyFuncUnknown :
* Execute One time , when an UNKNOWN person gets within the detection distance , he wil automatically be enrolled and affected an ID .

* This function triggers One time per user , after being added to the DB , and return its affected ID.
  * Message Shown on the Console : 
  
         [+] ==> UNKNOWN USER ADDED TO BD. 
         [+] AFFECTING ID : {} TO THE USER
     
## DummyFuncEnrolled :
* Execute whenever a known person gets within detection distance.

* This function triggers every time a known user is within detection distance , and returns its ID.
  * Message Shown on the Console : 
  
        [+] KNOWN USER.
        [+] AFFECTED ID IS : {}
     

