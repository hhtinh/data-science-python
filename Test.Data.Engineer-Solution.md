Technical Task Proposed Solution Summary
========================================

I would like to share a Proposed Solution for the Order Event Generation Task toghether with some other observations and thoughts:

## App Set-up and Execution Instruction

### Requirement: Python 3.6.8

* The App contains a Python script file `order-events.py`
* To run the file, the system must have Python 3.6.8 installed
* On **Windows**: it's recommended to install [Anaconda](https://www.anaconda.com/distribution/#download-section) and create an environment with Python 3.6.8 before executing the script:
    ```bash
    conda create -n py368 python=3.6.8
    conda activate py368
    ```

### Executing the script															

* The script requires 4 arguments:
    * number-of-orders - Number of orders to generate
    * batch-size - Number of events per file
    * interval - Interval in seconds between each file being created
    * output-directory - Output directory for all created files
* To run the app

    ```bash
    python order-events.py <number-of-orders> <batch-size> <interval> <output-directory>
    ```
    * For example:
    ```bash
    python order-events.py 27 10 3 orders
    ```
    - This command will create a total of 27 orders (each with 2 events associated with, either {OrderPlaced, OrderDelivered} or {OrderPlaced, OrderCancelled})
    - with the batch size of 10 orders (= 20 events) per file
    - with the interval of 3 seconds
    - and the output directory will be <orders> (within the same directory as the Python script)

## Other observations and thoughts

* Getting the correct UTC time is complicated in Python
* Datetime format in Python is also not simple as it should be
* I spent about 8 hours for this task
* If I had more time, I would add more validations for the input arguments of the scripts and make the App more interactive for users
* I choose Python because I think it's the most used programming language for Data Engineer at the moment
* Some brief about myself:
    ```json
    { "FullName": "Huỳnh Hữu Tính",
      "Email": "hhtinh@live.com",
      "TotalExperience": "16+ years",
      "ExperienceInData": "12+ years",
      "LinkedInProfile": "https://www.linkedin.com/in/tinhhuynh"
    }  
    ```
