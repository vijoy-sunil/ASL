#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <syslog.h>
#include <sstream>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

/// Maximum difference threshold to accept an input
#define DIFF_THRESH 230


#define HRES (320)
#define VRES (240)
#define NUM_THREADS (5)
#define NSEC (1000000000)
#define MSEC (1000000)
#define NSEC_PER_MICROSEC (1000)
#define NUM_CPUS (1)
#define KEY_ESC (27)

#define MAX_WORDS 26                                    // Number of letters
#define NUM_LAST_LETTERS 3                         // Number of letters to store
#define MIN_FREQ 2                                // Minimum frequency of last letters
#define THRESH 200
#define SAMPLE_RATE 1
#define RESET_THRESH  25000000


// POSIX thread declarations and scheduling attributes
pthread_t threads[NUM_THREADS];
pthread_attr_t rt_sched_attr[NUM_THREADS];

pthread_attr_t main_attr;
int rt_max_prio, rt_min_prio;

struct sched_param rt_param[NUM_THREADS];
struct sched_param main_param;
pid_t mainpid;

// initialize start time to 0sec and 0 nanosec

struct timespec start_time = {0, 0};

// Thread declarations
       
void *th1_captureimage(void*);     
void *th2_extracthand(void*);    
void *th3_extractfeature(void*);    
void *th4_identifyletter(void*);    
void *th5_displayletter(void*);    

void aslt_init(); 
//void doSystemCalls(char c);

// Global data

Mat rgb_image;                                // output th1--> rgb_imge
Mat binary_image;    
Mat drawing;                              // output th2--> binary_image
vector<vector<Point> > feature_image;                    // output th3--> feature_image
char asl_letter;                            // output th4--> asl_letter
VideoCapture capture;
Ptr<BackgroundSubtractor> pMOG2;
vector<Point> letters[MAX_WORDS];
vector<Vec4i> hierarchy;
vector<vector<Point> > contours;
int frames = 0;
int maxIndex = 0;
int reset = 0;

sem_t     SIGNDECODED_SM,
    NEWIMAGEREADY_SM,
    HANDREADY_SM,
    FEATUREREADY_SM,
    SIGNREADY_SM;


void print_scheduler(void)
{
   int schedType;

   schedType = sched_getscheduler(getpid());

   switch(schedType)
   {
     case SCHED_FIFO:
           printf("Pthread Policy is SCHED_FIFO\n");
           break;
     case SCHED_OTHER:
           printf("Pthread Policy is SCHED_OTHER\n");
       break;
     case SCHED_RR:
           printf("Pthread Policy is SCHED_OTHER\n");
           break;
     default:
       printf("Pthread Policy is UNKNOWN\n");
       break;
   }

}

int delta_t(struct timespec *stop, struct timespec *start, struct timespec *delta_t)
{
  int dt_sec=stop->tv_sec - start->tv_sec;
  int dt_nsec=stop->tv_nsec - start->tv_nsec;

  if(dt_sec >= 0)
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC+dt_nsec;
    }
  }
  else
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC+dt_nsec;
    }
  }

  return(1);
}

int distance_2(vector<Point> a, vector<Point> b) {
    int maxDistAB = 0;
    for (size_t i = 0; i < a.size(); i++) {
        int minB = 1000000;
        for (size_t j = 0; j < b.size(); j++) {
            int dx = (a[i].x - b[j].x);
            int dy = (a[i].y - b[j].y);
            int tmpDist = dx*dx + dy*dy;

            if (tmpDist < minB) {
                minB = tmpDist;
            }
            if (tmpDist == 0) {
                break; // can't get better than equal.
            }
        }
        maxDistAB += minB;
    }
    return maxDistAB;
}
double distance_hausdorff(vector<Point> a, vector<Point> b) {
    int maxDistAB = distance_2(a, b);
    int maxDistBA = distance_2(b, a);
    int maxDist = max(maxDistAB,maxDistBA);

    return sqrt((double)maxDist);
}


void aslt_init(void)
{
    int numframe = 0;
    Mat frame;
    sem_init(&SIGNDECODED_SM, 0, 1);
    sem_init(&NEWIMAGEREADY_SM, 0, 0);
    sem_init(&HANDREADY_SM, 0, 0);
    sem_init(&FEATUREREADY_SM, 0, 0);
    sem_init(&SIGNREADY_SM, 0, 0);

    //************Preload letter images starts*********//
    for (int i = 0; i < MAX_WORDS; i++) {
        char buf[13 * sizeof(char)];
        sprintf(buf, "images/%c.png", (char)('a' + i));
        Mat im = imread(buf, 1);
        if (im.data) {
            Mat bwim;
            cvtColor(im, bwim, CV_RGB2GRAY);
            Mat threshold_output;
                        // Detect edges using Threshold
            threshold( bwim, threshold_output, THRESH, 255, THRESH_BINARY );
            findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
            letters[i] = contours[0];
        }
    }
    //************Preload letter images ends*********//


    //************learn starts**********************//
    
    pMOG2 = new BackgroundSubtractorMOG2();
    //************learn ends  **********************//

}

void *th1_captureimage(void*)
{
    Mat frame; 
     capture = VideoCapture(0); 
                                       // current frame
    while(1)
    {
        printf("Thread #1: Capture Image\n\r");
        struct timespec start_time_th1 = {0, 0};
            struct timespec finish_time_th1 = {0, 0};
            struct timespec thread_dt_th1 = {0, 0};

        sem_wait(&SIGNDECODED_SM);                    // SIGNDECODED_SM take    
        clock_gettime(CLOCK_REALTIME, &start_time_th1);                            

                 // Create the capture object
        if (!capture.isOpened()) {
            cerr << "Cannot Open Webcam !!!" << endl;        // Error in opening the video input
            exit(EXIT_FAILURE);
        }
                                        // Read the current frame
        if (!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }                
        cv::Rect myROI(50, 150, 200, 200);                // Crop Frame to smaller region : output --> rgb_image
        rgb_image = frame(myROI);                    
        imshow("th1_captureimage", rgb_image);                // output th1--> rgb_imge
        char q = cvWaitKey(33);    
        
        clock_gettime(CLOCK_REALTIME, &finish_time_th1);
        delta_t(&finish_time_th1, &start_time_th1, &thread_dt_th1);    //compute the time of thread execution from the start and end times
        
        printf("\nThread #1 exec %lf msec \n", (double)((double)thread_dt_th1.tv_nsec / MSEC));
        sem_post(&NEWIMAGEREADY_SM);                    // NEWIMAGEREADY_SM give
    }
    capture.release();

} 

void *th2_extracthand(void*)
{
    while(1)
    {
        printf("Thread #2: Extract hand\n\r");
        struct timespec start_time_th2 = {0, 0};
            struct timespec finish_time_th2 = {0, 0};
            struct timespec thread_dt_th2 = {0, 0};

        sem_wait(&NEWIMAGEREADY_SM);                    // NEWIMAGEREADY_SM take
        clock_gettime(CLOCK_REALTIME, &start_time_th2);
        
        if(reset <= 10){
        reset++;
        pMOG2 = new BackgroundSubtractorMOG2();
        }


        pMOG2->operator()(rgb_image, binary_image);
        //imshow("raw", binary_image);
        
        clock_gettime(CLOCK_REALTIME, &finish_time_th2);
        delta_t(&finish_time_th2, &start_time_th2, &thread_dt_th2);//compute the time of thread execution from the start and end times
        
        printf("\nThread #2 exec %lf msec \n", (double)((double)thread_dt_th2.tv_nsec / MSEC));
        sem_post(&HANDREADY_SM);                    // HANDREADY_SM give
    }

} 

void *th3_extractfeature(void*)
{
    while(1)
    {
            
        Mat threshold_output;                            // Generate Convex Hull
            printf("Thread #3: Extract Feature\n\r");
        struct timespec start_time_th3 = {0, 0};
            struct timespec finish_time_th3 = {0, 0};
            struct timespec thread_dt_th3 = {0, 0};
        
        sem_wait(&HANDREADY_SM);                    // HANDREADY_SM take
        clock_gettime(CLOCK_REALTIME, &start_time_th3);
                                
        threshold( binary_image, threshold_output, THRESH, 255, THRESH_BINARY );            // Detect edges using Threshold

        findContours( threshold_output, feature_image, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );    // Find contours
        //imshow("feature" , feature_image);
        drawing = Mat::zeros(rgb_image.size(), CV_8UC3);                            // Find largest contour
        double largest_area = 0;
        for (int j = 0; j < feature_image.size(); j++) 
        {
            double area = contourArea(feature_image[j], false);    // Find the area of contour
            if (area > largest_area) {
                largest_area = area;
                maxIndex = j;                      // Store the index of largest contour
            }
        }
        
        //printf("%d", maxIndex);                                // Draw Largest Contours
        Scalar color = Scalar(0, 0, 255);
        drawContours(drawing, feature_image, maxIndex, Scalar(255, 255, 255), CV_FILLED); // fill white

                                        // Draw Contours
        Mat contourImg = Mat::zeros(rgb_image.size(), CV_8UC3);
        drawContours( contourImg, feature_image, maxIndex, Scalar(0, 0, 255), 2, 8, hierarchy, 0, Point(0, 0) );
        
        // Reset if too much noise
        Scalar sums = sum(drawing);
        int s = sums[0] + sums[1] + sums[2] + sums[3];
        if (s >= RESET_THRESH) {
            reset = 10;
        }

        imshow("Foreground", drawing);
        if (contourImg.rows > 0)
            imshow("th3_extractfeature", contourImg);
        char q = cvWaitKey(33);    
        
        clock_gettime(CLOCK_REALTIME, &finish_time_th3);
        delta_t(&finish_time_th3, &start_time_th3, &thread_dt_th3);    // compute the time of thread execution from the start and end times
        
        printf("\nThread #3 exec %lf msec \n", (double)((double)thread_dt_th3.tv_nsec / MSEC));
        sem_post(&FEATUREREADY_SM);                    // FEATUREREADY_SM give    
    }

}

void *th4_identifyletter(void*)
{
    while(1)
    {
        printf("Thread #4: Identify letter\n\r");
        struct timespec start_time_th4 = {0, 0};
            struct timespec finish_time_th4 = {0, 0};
            struct timespec thread_dt_th4 = {0, 0};

        
        sem_wait(&FEATUREREADY_SM);                    // FEATUREREADY_SM take
        clock_gettime(CLOCK_REALTIME, &start_time_th4);
        
        //************
        
        // Compare to reference images
        if (feature_image.size() > 0 && frames++ > SAMPLE_RATE && feature_image[maxIndex].size() >= 5) {
            RotatedRect testRect = fitEllipse(feature_image[maxIndex]);
            frames = 0;
            double lowestDiff = HUGE_VAL;
            for (int i = 0; i < MAX_WORDS; i++) {
                if (letters[i].size() == 0) continue;

                double diff = distance_hausdorff(letters[i], feature_image[maxIndex]);

                if (diff < lowestDiff) {
                    lowestDiff = diff;
                    asl_letter = 'a' + i;
                }
            }
            if (lowestDiff > DIFF_THRESH) { // Dust
                asl_letter = 0;
            }
            cout << asl_letter << " | diff: " << lowestDiff << endl;
            printf("| diff: %f \n\r",lowestDiff );
        }
        //************

        clock_gettime(CLOCK_REALTIME, &finish_time_th4);
        delta_t(&finish_time_th4, &start_time_th4, &thread_dt_th4);    // compute the time of thread execution from the start and end times
        
        printf("\nThread #4 exec %lf msec \n", (double)((double)thread_dt_th4.tv_nsec / MSEC));
        sem_post(&SIGNREADY_SM);                    // SIGNREADY_SM give
    }

}

void *th5_displayletter(void*)
{
    int letterCount = 0;                             // number of letters captured since last display
    char lastLetters[NUM_LAST_LETTERS] = {0};
    Mat letter_image = Mat::zeros(200, 200, CV_8UC3);        
    char lastExecLetter = 0;                          // last letter sent to doSystemCalls()

    while(1)
    {
        printf("Thread #5: Display output\n\r");
        struct timespec start_time_th5 = {0, 0};
            struct timespec finish_time_th5 = {0, 0};
            struct timespec thread_dt_th5 = {0, 0};
        
        sem_wait(&SIGNREADY_SM);                    // SIGNREADY_SM take
        clock_gettime(CLOCK_REALTIME, &start_time_th5);

        letterCount %= NUM_LAST_LETTERS;                // Show majority of last letters captured
        lastLetters[letterCount++] = asl_letter;            // input from th4
        letter_image = Mat::zeros(200, 200, CV_8UC3);

        int counts[MAX_WORDS+1] = {0};
        for (int i = 0; i < NUM_LAST_LETTERS; i++)
            counts[lastLetters[i] + 1 - 'a']++;

        int maxCount = 0;
        char maxChar = 0;
        for (int i = 0; i < MAX_WORDS+1; i++) {
            if (counts[i] > maxCount) {
                maxCount = counts[i];
                maxChar = i;
            }
        }

        if (maxChar && maxCount >= MIN_FREQ) 
        {
            maxChar = maxChar - 1 + 'a';
            char buf[2 * sizeof(char)];
            sprintf(buf, "%c", maxChar);

            putText(letter_image, buf, Point(10, 75), CV_FONT_NORMAL, 3, Scalar(255, 255, 255), 1, 1);
            vector<vector<Point> > dummy;
            dummy.push_back(letters[maxChar-'a']);

            drawContours( letter_image, dummy, 0, Scalar(255, 0, 0), 2, 8, hierarchy, 0, Point(0, 0) );
            if (maxChar != lastExecLetter) {
                lastExecLetter = maxChar;
                //doSystemCalls(maxChar);
            }
        }
        imshow("Letter", letter_image);                    // output th5--> letter_image    
        char q = cvWaitKey(33);    
        
        clock_gettime(CLOCK_REALTIME, &finish_time_th5);
        delta_t(&finish_time_th5, &start_time_th5, &thread_dt_th5);    //compute the time of thread execution from the start and end times
        
        printf("\nThread #5 exec %lf msec \n", (double)((double)thread_dt_th5.tv_nsec / MSEC));
        sem_post(&SIGNDECODED_SM);                    // SIGNDECODED_SM give    
    }

}

int main( int argc, char** argv )
{
    int rc,scope,i;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);                            //setting the CPU cores of all cores to 0.
    for(i=0; i < NUM_CPUS; i++)
    CPU_SET(i, &cpuset);
    char keyboard = 0;    // last key pressed
    int training_mode = 0; // 0 = no training; 1 = training
    printf("Do you want to train? \n\r");
          cin >> keyboard;
          if(keyboard == 'y')
        {
        training_mode = 1;
        }


    if(training_mode)
    {
    capture = VideoCapture(0);
    pMOG2 = new BackgroundSubtractorMOG2();
    while (keyboard != KEY_ESC) {
        printf("inside training \n\r ");
        if (!capture.isOpened()) {
        // Error in opening the video input
        cerr << "Cannot Open Webcam... " << endl;
        exit(EXIT_FAILURE);
        }

            Mat frame;           // current frame
            Mat fgMaskMOG2;      // fg mask fg mask generated by MOG2 method
        if (!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }

        // Crop Frame to smaller region
        cv::Rect myROI(50, 150, 200, 200);
        Mat cropFrame = frame(myROI);

        // Update the background model
        pMOG2->operator()(cropFrame, fgMaskMOG2);

        // Generate Convex Hull
        Mat threshold_output;
        vector<vector<Point> > contours;

        // Detect edges using Threshold
        threshold(fgMaskMOG2 , threshold_output, THRESH, 255, THRESH_BINARY );

        // Find contours
        findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        // Find largest contour
        Mat drawing1 = Mat::zeros(cropFrame.size(), CV_8UC3);
        double largest_area = 0;
        int maxIndex = 0;
        for (int j = 0; j < contours.size(); j++) {
            double area = contourArea(contours[j], false);    // Find the area of contour
            if (area > largest_area) {
                largest_area = area;
                maxIndex = j;  // Store the index of largest contour
            }
        }

        // Draw Largest Contours
        Scalar color = Scalar(0, 0, 255);
        drawContours(drawing1, contours, maxIndex, Scalar(255, 255, 255), CV_FILLED); // fill white
        // Draw Contours
        Mat contourImg = Mat::zeros(cropFrame.size(), CV_8UC3);
        drawContours( contourImg, contours, maxIndex, Scalar(0, 0, 255), 2, 8, hierarchy, 0, Point(0, 0) );

        // Reset if too much noise
        Scalar sums = sum(drawing1);
        int s = sums[0] + sums[1] + sums[2] + sums[3];
        if (s >= RESET_THRESH) {
            pMOG2 = new BackgroundSubtractorMOG2();
            continue;
        }

        // Show the current frame and the fg masks
        imshow("Crop Frame", cropFrame);
        imshow("Foreground", drawing1);
        if (contourImg.rows > 0)
            imshow("Contour", contourImg);
        //har ch = cvWaitKey(33);
        
        keyboard = waitKey(1);
        
        if (keyboard >= 'a' && keyboard <= 'z') {
            cout << "Wrote letter '" << (char)keyboard << '\'' << endl;

            // save in memory
            letters[keyboard - 'a'] = contours[maxIndex];

            // write to file
            char buf[13 * sizeof(char)];
            sprintf(buf, "images/%c.png", (char)keyboard);
            imwrite(buf, drawing1);
        }

        // Manual reset
        if (keyboard == ' ')
            pMOG2 = new BackgroundSubtractorMOG2();

    // Delete capture object

    }
    destroyAllWindows();
    capture.release();
}


    mainpid=getpid();                            //get the thread id of the calling thread.

    rt_max_prio = sched_get_priority_max(SCHED_FIFO);            //max priority of the SCHED_FIFO
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);            //min priority of the SCHED_FIFO

    printf("\nBefore Adjustments to Schedule Policy:");
    print_scheduler();                             //print scheduler before assigning SCHED_FIFO

    rc=sched_getparam(mainpid, &main_param);                //get the scheduling parameters of the thread and transferring it to main_param
    main_param.sched_priority=rt_max_prio;                    //setting the max priority of the calling thread to 99

    rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);        //set the SCHED_FIFO scheduler of the main_param.
    if(rc < 0) perror("main_param");

    printf("\nAfter Adjustments to Schedule Policy:");
    print_scheduler();                                //print scheduler after assigning SCHED_FIFO

    pthread_attr_getscope(&main_attr, &scope);                //obtain the scope of the main_attr and print it
    if(scope == PTHREAD_SCOPE_SYSTEM)
        printf("PTHREAD SCOPE SYSTEM\n");
    else if (scope == PTHREAD_SCOPE_PROCESS)
         printf("PTHREAD SCOPE PROCESS\n");
    else
         printf("PTHREAD SCOPE UNKNOWN\n");

                                        //Attribute settings for the Threads
       for(i=0; i <NUM_THREADS; i++)
       {
        rc=pthread_attr_init(&rt_sched_attr[i]);                    // intializing the pthread attributes for the five threads.
        rc=pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);    // set to explicit schedule policy and later to SCHED_FIFO
        rc=pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);            //set the schedule policy of the five threads to SCHED_FIFO
        rc=pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &cpuset);    //set the affinity of the CPU cores to zero

        rt_param[i].sched_priority=rt_max_prio-i;                    //set the priorities of the hreads as 98,97,96,95 and 94 respectively.
        pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);            //set the scheduling parameters of the five pthreads.
       }

    printf("rt_max_prio=%d\n", rt_max_prio);
    printf("rt_min_prio=%d\n", rt_min_prio);

    aslt_init();

    printf("threads spawning\r\n");
    pthread_create(&threads[0],                   // pointer to thread descriptor
                    &rt_sched_attr[0],      // use set attributes
                    th1_captureimage,     // thread function entry point
                    (void *)(NULL)         // parameters to pass in
                    );

    pthread_create(&threads[1],               
                     &rt_sched_attr[1],     
                     th2_extracthand,         
                     (void *)(NULL)         
                    );
    pthread_create(&threads[2],               
                     &rt_sched_attr[2],     
                     th3_extractfeature,     
                     (void *)(NULL)         
                    );
    pthread_create(&threads[3],               
                     &rt_sched_attr[3],    
                     th4_identifyletter,     
                     (void *)(NULL)         
                    );
    pthread_create(&threads[4],               
                     &rt_sched_attr[4],    
                     th5_displayletter,     
                     (void *)(NULL)         
                    );

    for(i=0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);                //join the pthread wiith the existing processes

    pthread_exit(NULL);
    capture.release();                        // Delete capture object
     
}