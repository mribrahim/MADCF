/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef __MYCSRT_TRACKER_HPP__
#define __MYCSRT_TRACKER_HPP__

#define MYCV_UNUSED(name) (void)name

#include "opencv2/core.hpp"
#include "feature.hpp"
#include "trackerCSRTSegmentation.hpp"
#include "trackerCSRTScaleEstimation.hpp"

/*
 * Partially based on:
 * ====================================================================================================================
 *   - [AAM] S. Salti, A. Cavallaro, L. Di Stefano, Adaptive Appearance Modeling for Video Tracking: Survey and Evaluation
 *  - [AMVOT] X. Li, W. Hu, C. Shen, Z. Zhang, A. Dick, A. van den Hengel, A Survey of Appearance Models in Visual Object Tracking
 *
 * This Tracking API has been designed with PlantUML. If you modify this API please change UML files under modules/tracking/doc/uml
 *
 */

namespace myCSRT
{

  //! @addtogroup tracking
  //! @{

  /************************************ TrackerFeature Base Classes ************************************/

  /** @brief Abstract base class for TrackerFeature that represents the feature.
   */
  class TrackerFeature
  {
  public:
    virtual ~TrackerFeature();

    /** @brief Compute the features in the images collection
      @param images The images
      @param response The output response
       */
    void compute(const std::vector<cv::Mat> &images, cv::Mat &response);

    /** @brief Create TrackerFeature by tracker feature type
      @param trackerFeatureType The TrackerFeature name

      The modes available now:

      -   "HAAR" -- Haar Feature-based

      The modes that will be available soon:

      -   "HOG" -- Histogram of Oriented Gradients features
      -   "LBP" -- Local Binary Pattern features
      -   "FEATURE2D" -- All types of Feature2D
       */
    static cv::Ptr<TrackerFeature> create(const cv::String &trackerFeatureType);

    /** @brief Identify most effective features
      @param response Collection of response for the specific TrackerFeature
      @param npoints Max number of features

      @note This method modifies the response parameter
       */
    virtual void selection(cv::Mat &response, int npoints) = 0;

    /** @brief Get the name of the specific TrackerFeature
     */
    cv::String getClassName() const;

  protected:
    virtual bool computeImpl(const std::vector<cv::Mat> &images, cv::Mat &response) = 0;

    cv::String className;
  };

  /** @brief Class that manages the extraction and selection of features

  @cite AAM Feature Extraction and Feature Set Refinement (Feature Processing and Feature Selection).
  See table I and section III C @cite AMVOT Appearance modelling -\> Visual representation (Table II,
  section 3.1 - 3.2)

  TrackerFeatureSet is an aggregation of TrackerFeature

  @sa
     TrackerFeature

   */
  class TrackerFeatureSet
  {
  public:
    TrackerFeatureSet();

    ~TrackerFeatureSet();

    /** @brief Extract features from the images collection
      @param images The input images
       */
    void extraction(const std::vector<cv::Mat> &images);

    /** @brief Identify most effective features for all feature types (optional)
     */
    void selection();

    /** @brief Remove outliers for all feature types (optional)
     */
    void removeOutliers();

    /** @brief Add TrackerFeature in the collection. Return true if TrackerFeature is added, false otherwise
      @param trackerFeatureType The TrackerFeature name

      The modes available now:

      -   "HAAR" -- Haar Feature-based

      The modes that will be available soon:

      -   "HOG" -- Histogram of Oriented Gradients features
      -   "LBP" -- Local Binary Pattern features
      -   "FEATURE2D" -- All types of Feature2D

      Example TrackerFeatureSet::addTrackerFeature : :
      @code
          //sample usage:

          Ptr<TrackerFeature> trackerFeature = new TrackerFeatureHAAR( HAARparameters );
          featureSet->addTrackerFeature( trackerFeature );

          //or add CSC sampler with default parameters
          //featureSet->addTrackerFeature( "HAAR" );
      @endcode
      @note If you use the second method, you must initialize the TrackerFeature
       */
    bool addTrackerFeature(cv::String trackerFeatureType);

    /** @overload
      @param feature The TrackerFeature class
      */
    bool addTrackerFeature(cv::Ptr<TrackerFeature> &feature);

    /** @brief Get the TrackerFeature collection (TrackerFeature name, TrackerFeature pointer)
     */
    const std::vector<std::pair<cv::String, cv::Ptr<TrackerFeature>>> &getTrackerFeature() const;

    /** @brief Get the responses

      @note Be sure to call extraction before getResponses Example TrackerFeatureSet::getResponses : :
       */
    const std::vector<cv::Mat> &getResponses() const;

  private:
    void clearResponses();
    bool blockAddTrackerFeature;

    std::vector<std::pair<cv::String, cv::Ptr<TrackerFeature>>> features; // list of features
    std::vector<cv::Mat> responses;                                       // list of response after compute
  };

  /************************************ TrackerSampler Base Classes ************************************/

  /** @brief Abstract base class for TrackerSamplerAlgorithm that represents the algorithm for the specific
  sampler.
   */
  class TrackerSamplerAlgorithm
  {
  public:
    /**
     * \brief Destructor
     */
    virtual ~TrackerSamplerAlgorithm();

    /** @brief Create TrackerSamplerAlgorithm by tracker sampler type.
      @param trackerSamplerType The trackerSamplerType name

      The modes available now:

      -   "CSC" -- Current State Center
      -   "CS" -- Current State
       */
    static cv::Ptr<TrackerSamplerAlgorithm> create(const cv::String &trackerSamplerType);

    /** @brief Computes the regions starting from a position in an image.

      Return true if samples are computed, false otherwise

      @param image The current frame
      @param boundingBox The bounding box from which regions can be calculated

      @param sample The computed samples @cite AAM Fig. 1 variable Sk
       */
    bool sampling(const cv::Mat &image, cv::Rect boundingBox, std::vector<cv::Mat> &sample);

    /** @brief Get the name of the specific TrackerSamplerAlgorithm
     */
    cv::String getClassName() const;

  protected:
    cv::String className;

    virtual bool samplingImpl(const cv::Mat &image, cv::Rect boundingBox, std::vector<cv::Mat> &sample) = 0;
  };

  /**
   * \brief Class that manages the sampler in order to select regions for the update the model of the tracker
   * [AAM] Sampling e Labeling. See table I and section III B
   */

  /** @brief Class that manages the sampler in order to select regions for the update the model of the tracker

  @cite AAM Sampling e Labeling. See table I and section III B

  TrackerSampler is an aggregation of TrackerSamplerAlgorithm
  @sa
     TrackerSamplerAlgorithm
   */
  class TrackerSampler
  {
  public:
    /**
     * \brief Constructor
     */
    TrackerSampler();

    /**
     * \brief Destructor
     */
    ~TrackerSampler();

    /** @brief Computes the regions starting from a position in an image
      @param image The current frame
      @param boundingBox The bounding box from which regions can be calculated
       */
    void sampling(const cv::Mat &image, cv::Rect boundingBox);

    /** @brief Return the collection of the TrackerSamplerAlgorithm
     */
    const std::vector<std::pair<cv::String, cv::Ptr<TrackerSamplerAlgorithm>>> &getSamplers() const;

    /** @brief Return the samples from all TrackerSamplerAlgorithm, @cite AAM Fig. 1 variable Sk
     */
    const std::vector<cv::Mat> &getSamples() const;

    /** @brief Add TrackerSamplerAlgorithm in the collection. Return true if sampler is added, false otherwise
      @param trackerSamplerAlgorithmType The TrackerSamplerAlgorithm name

      The modes available now:
      -   "CSC" -- Current State Center
      -   "CS" -- Current State
      -   "PF" -- Particle Filtering

      Example TrackerSamplerAlgorithm::addTrackerSamplerAlgorithm : :
      @code
           TrackerSamplerCSC::Params CSCparameters;
           Ptr<TrackerSamplerAlgorithm> CSCSampler = new TrackerSamplerCSC( CSCparameters );

           if( !sampler->addTrackerSamplerAlgorithm( CSCSampler ) )
             return false;

           //or add CSC sampler with default parameters
           //sampler->addTrackerSamplerAlgorithm( "CSC" );
      @endcode
      @note If you use the second method, you must initialize the TrackerSamplerAlgorithm
       */
    bool addTrackerSamplerAlgorithm(cv::String trackerSamplerAlgorithmType);

    /** @overload
      @param sampler The TrackerSamplerAlgorithm
      */
    bool addTrackerSamplerAlgorithm(cv::Ptr<TrackerSamplerAlgorithm> &sampler);

  private:
    std::vector<std::pair<cv::String, cv::Ptr<TrackerSamplerAlgorithm>>> samplers;
    std::vector<cv::Mat> samples;
    bool blockAddTrackerSampler;

    void clearSamples();
  };

  /************************************ TrackerModel Base Classes ************************************/

  /** @brief Abstract base class for TrackerTargetState that represents a possible state of the target.

  See @cite AAM \f$\hat{x}^{i}_{k}\f$ all the states candidates.

  Inherits this class with your Target state, In own implementation you can add scale variation,
  width, height, orientation, etc.
   */
  class TrackerTargetState
  {
  public:
    virtual ~TrackerTargetState(){};
    /**
     * \brief Get the position
     * \return The position
     */
    cv::Point2f getTargetPosition() const;

    /**
     * \brief Set the position
     * \param position The position
     */
    void setTargetPosition(const cv::Point2f &position);
    /**
     * \brief Get the width of the target
     * \return The width of the target
     */
    int getTargetWidth() const;

    /**
     * \brief Set the width of the target
     * \param width The width of the target
     */
    void setTargetWidth(int width);
    /**
     * \brief Get the height of the target
     * \return The height of the target
     */
    int getTargetHeight() const;

    /**
     * \brief Set the height of the target
     * \param height The height of the target
     */
    void setTargetHeight(int height);

  protected:
    cv::Point2f targetPosition;
    int targetWidth;
    int targetHeight;
  };

  /** @brief Represents the model of the target at frame \f$k\f$ (all states and scores)

  See @cite AAM The set of the pair \f$\langle \hat{x}^{i}_{k}, C^{i}_{k} \rangle\f$
  @sa TrackerTargetState
   */
  typedef std::vector<std::pair<cv::Ptr<TrackerTargetState>, float>> ConfidenceMap;

  /** @brief Represents the estimate states for all frames

  @cite AAM \f$x_{k}\f$ is the trajectory of the target up to time \f$k\f$

  @sa TrackerTargetState
   */
  typedef std::vector<cv::Ptr<TrackerTargetState>> Trajectory;

  /** @brief Abstract base class for TrackerStateEstimator that estimates the most likely target state.

  See @cite AAM State estimator

  See @cite AMVOT Statistical modeling (Fig. 3), Table III (generative) - IV (discriminative) - V (hybrid)
   */
  class TrackerStateEstimator
  {
  public:
    virtual ~TrackerStateEstimator();

    /** @brief Estimate the most likely target state, return the estimated state
      @param confidenceMaps The overall appearance model as a list of :cConfidenceMap
       */
    cv::Ptr<TrackerTargetState> estimate(const std::vector<ConfidenceMap> &confidenceMaps);

    /** @brief Update the ConfidenceMap with the scores
      @param confidenceMaps The overall appearance model as a list of :cConfidenceMap
       */
    void update(std::vector<ConfidenceMap> &confidenceMaps);

    /** @brief Create TrackerStateEstimator by tracker state estimator type
      @param trackeStateEstimatorType The TrackerStateEstimator name

      The modes available now:

      -   "BOOSTING" -- Boosting-based discriminative appearance models. See @cite AMVOT section 4.4

      The modes available soon:

      -   "SVM" -- SVM-based discriminative appearance models. See @cite AMVOT section 4.5
       */
    static cv::Ptr<TrackerStateEstimator> create(const cv::String &trackeStateEstimatorType);

    /** @brief Get the name of the specific TrackerStateEstimator
     */
    cv::String getClassName() const;

  protected:
    virtual cv::Ptr<TrackerTargetState> estimateImpl(const std::vector<ConfidenceMap> &confidenceMaps) = 0;
    virtual void updateImpl(std::vector<ConfidenceMap> &confidenceMaps) = 0;
    cv::String className;
  };

  class TrackerStateEstimatorSVM : public TrackerStateEstimator
  {
  public:
    TrackerStateEstimatorSVM();
    ~TrackerStateEstimatorSVM();

  protected:
    cv::Ptr<TrackerTargetState> estimateImpl(const std::vector<ConfidenceMap> &confidenceMaps) CV_OVERRIDE;
    void updateImpl(std::vector<ConfidenceMap> &confidenceMaps) CV_OVERRIDE;
  };

  class TrackerModel
  {
  public:
    /**
     * \brief Constructor
     */
    TrackerModel();

    /**
     * \brief Destructor
     */
    virtual ~TrackerModel();

    /** @brief Set TrackerEstimator, return true if the tracker state estimator is added, false otherwise
      @param trackerStateEstimator The TrackerStateEstimator
      @note You can add only one TrackerStateEstimator
       */
    bool setTrackerStateEstimator(cv::Ptr<TrackerStateEstimator> trackerStateEstimator);

    /** @brief Estimate the most likely target location

      @cite AAM ME, Model Estimation table I
      @param responses Features extracted from TrackerFeatureSet
       */
    void modelEstimation(const std::vector<cv::Mat> &responses);

    /** @brief Update the model

      @cite AAM MU, Model Update table I
       */
    void modelUpdate();

    /** @brief Run the TrackerStateEstimator, return true if is possible to estimate a new state, false otherwise
     */
    bool runStateEstimator();

    /** @brief Set the current TrackerTargetState in the Trajectory
      @param lastTargetState The current TrackerTargetState
       */
    void setLastTargetState(const cv::Ptr<TrackerTargetState> &lastTargetState);

    /** @brief Get the last TrackerTargetState from Trajectory
     */
    cv::Ptr<TrackerTargetState> getLastTargetState() const;

    /** @brief Get the list of the ConfidenceMap
     */
    const std::vector<ConfidenceMap> &getConfidenceMaps() const;

    /** @brief Get the last ConfidenceMap for the current frame
     */
    const ConfidenceMap &getLastConfidenceMap() const;

    /** @brief Get the TrackerStateEstimator
     */
    cv::Ptr<TrackerStateEstimator> getTrackerStateEstimator() const;

  private:
    void clearCurrentConfidenceMap();

  protected:
    std::vector<ConfidenceMap> confidenceMaps;
    cv::Ptr<TrackerStateEstimator> stateEstimator;
    ConfidenceMap currentConfidenceMap;
    Trajectory trajectory;
    int maxCMLength;

    virtual void modelEstimationImpl(const std::vector<cv::Mat> &responses) = 0;
    virtual void modelUpdateImpl() = 0;
  };

  /************************************ Tracker Base Class ************************************/

  class Tracker //: public virtual Algorithm
  {
  public:
    virtual ~Tracker(); // CV_OVERRIDE;

    /** @brief Initialize the tracker with a known bounding box that surrounded the target
      @param image The initial frame
      @param boundingBox The initial bounding box

      @return True if initialization went succesfully, false otherwise
       */
    bool init(cv::InputArray image, const cv::Rect2d &boundingBox);

    /** @brief Update the tracker, find the new most likely bounding box for the target
      @param image The current frame
      @param boundingBox The bounding box that represent the new target location, if true was returned, not
      modified otherwise

      @return True means that target was located and false means that tracker cannot locate target in
      current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed
      missing from the frame (say, out of sight)
       */
    bool update(cv::InputArray image, cv::Rect2d &boundingBox, const cv::Mat &homoMat);

    virtual void read(const cv::FileNode &fn); // cv::CV_OVERRIDE = 0;
    virtual void write(cv::FileStorage &fs);   // const CV_OVERRIDE = 0;

  protected:
    virtual bool initImpl(const cv::Mat &image, const cv::Rect2d &boundingBox) = 0;
    virtual bool updateImpl(const cv::Mat &image, cv::Rect2d &boundingBox, const cv::Mat &homoMat) = 0;

    bool isInit;

    cv::Ptr<TrackerFeatureSet> featureSet;
    cv::Ptr<TrackerSampler> sampler;
    cv::Ptr<TrackerModel> model;
  };

  /** @brief TrackerSampler based on CSC (current state centered), used by MIL algorithm TrackerMIL
   */
  class TrackerSamplerCSC : public TrackerSamplerAlgorithm
  {
  public:
    enum
    {
      MODE_INIT_POS = 1,  //!< mode for init positive samples
      MODE_INIT_NEG = 2,  //!< mode for init negative samples
      MODE_TRACK_POS = 3, //!< mode for update positive samples
      MODE_TRACK_NEG = 4, //!< mode for update negative samples
      MODE_DETECT = 5     //!< mode for detect samples
    };

    struct Params
    {
      Params();
      float initInRad;     //!< radius for gathering positive instances during init
      float trackInPosRad; //!< radius for gathering positive instances during tracking
      float searchWinSize; //!< size of search window
      int initMaxNegNum;   //!< # negative samples to use during init
      int trackMaxPosNum;  //!< # positive samples to use during training
      int trackMaxNegNum;  //!< # negative samples to use during training
    };

    /** @brief Constructor
      @param parameters TrackerSamplerCSC parameters TrackerSamplerCSC::Params
       */
    TrackerSamplerCSC(const TrackerSamplerCSC::Params &parameters = TrackerSamplerCSC::Params());

    /** @brief Set the sampling mode of TrackerSamplerCSC
      @param samplingMode The sampling mode

      The modes are:

      -   "MODE_INIT_POS = 1" -- for the positive sampling in initialization step
      -   "MODE_INIT_NEG = 2" -- for the negative sampling in initialization step
      -   "MODE_TRACK_POS = 3" -- for the positive sampling in update step
      -   "MODE_TRACK_NEG = 4" -- for the negative sampling in update step
      -   "MODE_DETECT = 5" -- for the sampling in detection step
       */
    void setMode(int samplingMode);

    ~TrackerSamplerCSC();

  protected:
    bool samplingImpl(const cv::Mat &image, cv::Rect boundingBox, std::vector<cv::Mat> &sample); // cv::CV_OVERRIDE;

  private:
    Params params;
    int mode;
    cv::RNG rng;

    std::vector<cv::Mat> sampleImage(const cv::Mat &img, int x, int y, int w, int h, float inrad, float outrad = 0, int maxnum = 1000000);
  };

  /** @brief TrackerSampler based on CS (current state), used by algorithm TrackerBoosting
   */
  class TrackerSamplerCS : public TrackerSamplerAlgorithm
  {
  public:
    enum
    {
      MODE_POSITIVE = 1, //!< mode for positive samples
      MODE_NEGATIVE = 2, //!< mode for negative samples
      MODE_CLASSIFY = 3  //!< mode for classify samples
    };

    struct Params
    {
      Params();
      float overlap;      //!< overlapping for the search windows
      float searchFactor; //!< search region parameter
    };
    /** @brief Constructor
      @param parameters TrackerSamplerCS parameters TrackerSamplerCS::Params
       */
    TrackerSamplerCS(const TrackerSamplerCS::Params &parameters = TrackerSamplerCS::Params());

    /** @brief Set the sampling mode of TrackerSamplerCS
      @param samplingMode The sampling mode

      The modes are:

      -   "MODE_POSITIVE = 1" -- for the positive sampling
      -   "MODE_NEGATIVE = 2" -- for the negative sampling
      -   "MODE_CLASSIFY = 3" -- for the sampling in classification step
       */
    void setMode(int samplingMode);

    ~TrackerSamplerCS();

    bool samplingImpl(const cv::Mat &image, cv::Rect boundingBox, std::vector<cv::Mat> &sample); // CV_OVERRIDE;
    cv::Rect getROI() const;

  private:
    cv::Rect getTrackingROI(float searchFactor);
    cv::Rect RectMultiply(const cv::Rect &rect, float f);
    std::vector<cv::Mat> patchesRegularScan(const cv::Mat &image, cv::Rect trackingROI, cv::Size patchSize);
    void setCheckedROI(cv::Rect imageROI);

    Params params;
    int mode;
    cv::Rect trackedPatch;
    cv::Rect validROI;
    cv::Rect ROI;
  };

  /** @brief This sampler is based on particle filtering.

  In principle, it can be thought of as performing some sort of optimization (and indeed, this
  tracker uses opencv's optim module), where tracker seeks to find the rectangle in given frame,
  which is the most *"similar"* to the initial rectangle (the one, given through the constructor).

  The optimization performed is stochastic and somehow resembles genetic algorithms, where on each new
  image received (submitted via TrackerSamplerPF::sampling()) we start with the region bounded by
  boundingBox, then generate several "perturbed" boxes, take the ones most similar to the original.
  This selection round is repeated several times. At the end, we hope that only the most promising box
  remaining, and these are combined to produce the subrectangle of image, which is put as a sole
  element in array sample.

  It should be noted, that the definition of "similarity" between two rectangles is based on comparing
  their histograms. As experiments show, tracker is *not* very succesfull if target is assumed to
  strongly change its dimensions.
   */
  class TrackerSamplerPF : public TrackerSamplerAlgorithm
  {
  public:
    /** @brief This structure contains all the parameters that can be varied during the course of sampling
      algorithm. Below is the structure exposed, together with its members briefly explained with
      reference to the above discussion on algorithm's working.
   */
    struct Params
    {
      Params();
      int iterationNum;     //!< number of selection rounds
      int particlesNum;     //!< number of "perturbed" boxes on each round
      double alpha;         //!< with each new round we exponentially decrease the amount of "perturbing" we allow (like in simulated annealing)
                            //!< and this very alpha controls how fast annealing happens, ie. how fast perturbing decreases
      cv::Mat_<double> std; //!< initial values for perturbing (1-by-4 array, as each rectangle is given by 4 values -- coordinates of opposite vertices,
                            //!< hence we have 4 values to perturb)
    };
    /** @brief Constructor
      @param chosenRect Initial rectangle, that is supposed to contain target we'd like to track.
      @param parameters
       */
    TrackerSamplerPF(const cv::Mat &chosenRect, const TrackerSamplerPF::Params &parameters = TrackerSamplerPF::Params());

  protected:
    bool samplingImpl(const cv::Mat &image, cv::Rect boundingBox, std::vector<cv::Mat> &sample); // CV_OVERRIDE;
  private:
    Params params;
    cv::Ptr<cv::MinProblemSolver> _solver;
    cv::Ptr<cv::MinProblemSolver::Function> _function;
  };

  /************************************ Specific TrackerFeature Classes ************************************/

  /**
   * \brief TrackerFeature based on Feature2D
   */
  class TrackerFeatureFeature2d : public TrackerFeature
  {
  public:
    /**
     * \brief Constructor
     * \param detectorType string of FeatureDetector
     * \param descriptorType string of DescriptorExtractor
     */
    TrackerFeatureFeature2d(cv::String detectorType, cv::String descriptorType);

    ~TrackerFeatureFeature2d(); // CV_OVERRIDE;

    void selection(cv::Mat &response, int npoints); // CV_OVERRIDE;

  protected:
    bool computeImpl(const std::vector<cv::Mat> &images, cv::Mat &response); // CV_OVERRIDE;

  private:
    std::vector<cv::KeyPoint> keypoints;
  };

  /**
   * \brief TrackerFeature based on HOG
   */
  class TrackerFeatureHOG : public TrackerFeature
  {
  public:
    TrackerFeatureHOG();

    ~TrackerFeatureHOG() CV_OVERRIDE;

    void selection(cv::Mat &response, int npoints); // cv::CV_OVERRIDE;

  protected:
    bool computeImpl(const std::vector<cv::Mat> &images, cv::Mat &response); // cv::CV_OVERRIDE;
  };

  /** @brief TrackerFeature based on HAAR features, used by TrackerMIL and many others algorithms
  @note HAAR features implementation is copied from apps/traincascade and modified according to MIL
   */
  class TrackerFeatureHAAR : public TrackerFeature
  {
  public:
    struct Params
    {
      Params();
      int numFeatures;   //!< # of rects
      cv::Size rectSize; //!< rect size
      bool isIntegral;   //!< true if input images are integral, false otherwise
    };

    /** @brief Constructor
      @param parameters TrackerFeatureHAAR parameters TrackerFeatureHAAR::Params
       */
    TrackerFeatureHAAR(const TrackerFeatureHAAR::Params &parameters = TrackerFeatureHAAR::Params());

    ~TrackerFeatureHAAR(); // CV_OVERRIDE;

    /** @brief Compute the features only for the selected indices in the images collection
      @param selFeatures indices of selected features
      @param images The images
      @param response Collection of response for the specific TrackerFeature
       */
    bool extractSelected(const std::vector<int> selFeatures, const std::vector<cv::Mat> &images, cv::Mat &response);

    /** @brief Identify most effective features
      @param response Collection of response for the specific TrackerFeature
      @param npoints Max number of features

      @note This method modifies the response parameter
       */
    void selection(cv::Mat &response, int npoints); // CV_OVERRIDE;

    /** @brief Swap the feature in position source with the feature in position target
    @param source The source position
    @param target The target position
   */
    bool swapFeature(int source, int target);

    /** @brief   Swap the feature in position id with the feature input
    @param id The position
    @param feature The feature
   */
    bool swapFeature(int id, myCSRT::CvHaarEvaluator::FeatureHaar &feature);

    /** @brief Get the feature in position id
      @param id The position
       */
    CvHaarEvaluator::FeatureHaar &getFeatureAt(int id);

  protected:
    bool computeImpl(const std::vector<cv::Mat> &images, cv::Mat &response); // CV_OVERRIDE;

  private:
    Params params;
    cv::Ptr<CvHaarEvaluator> featureEvaluator;
  };

  /**
   * \brief TrackerFeature based on LBP
   */
  class TrackerFeatureLBP : public TrackerFeature
  {
  public:
    TrackerFeatureLBP();

    ~TrackerFeatureLBP();

    void selection(cv::Mat &response, int npoints); // CV_OVERRIDE;

  protected:
    bool computeImpl(const std::vector<cv::Mat> &images, cv::Mat &response); // CV_OVERRIDE;
  };
  //
  // class MyTrackerCSRT : public Tracker
  //{
  // public:
  //  struct Params
  //  {
  //    /**
  //    * \brief Constructor
  //    */
  //    Params();
  //
  //    /**
  //    * \brief Read parameters from a file
  //    */
  //    void read(const cv::FileNode& /*fn*/);
  //
  //    /**
  //    * \brief
  //     *
  //     * ite parameters to a file
  //    */
  //    void write(cv::FileStorage& fs) const;
  //
  //    bool use_hog;
  //    bool use_color_names;
  //    bool use_gray;
  //    bool use_rgb;
  //    bool use_channel_weights;
  //    bool use_segmentation;
  //
  //    std::string window_function; //!<  Window function: "hann", "cheb", "kaiser"
  //    float kaiser_alpha;
  //    float cheb_attenuation;
  //
  //    float template_size;
  //    float gsl_sigma;
  //    float hog_orientations;
  //    float hog_clip;
  //    float padding;
  //    float filter_lr;
  //    float weights_lr;
  //    int num_hog_channels_used;
  //    int admm_iterations;
  //    int histogram_bins;
  //    float histogram_lr;
  //    int background_ratio;
  //    int number_of_scales;
  //    float scale_sigma_factor;
  //    float scale_model_max_area;
  //    float scale_lr;
  //    float scale_step;
  //
  //    float psr_threshold; //!< we lost the target, if the psr is lower than this.
  //  };
  //
  //  /** @brief Constructor
  //  @param parameters CSRT parameters MyTrackerCSRT::Params
  //  */
  //  static cv::Ptr<MyTrackerCSRT> create(const MyTrackerCSRT::Params &parameters);
  //
  //  static cv::Ptr<MyTrackerCSRT> create();
  //
  //  virtual void setInitialMask(cv::InputArray mask) = 0;
  //
  //  virtual ~MyTrackerCSRT();// CV_OVERRIDE {}
  //};

  class TrackerCSRTImpl //: public MyTrackerCSRT
  {
  public:
    struct Params
    {
      /**
       * \brief Constructor
       */
      Params();

      /**
       * \brief Read parameters from a file
       */
      void read(const cv::FileNode & /*fn*/);

      /**
       * \brief
       *
       * ite parameters to a file
       */
      void write(cv::FileStorage &fs) const;

      bool use_cnn_torch;
      bool use_hog;
      bool use_color_names;
      bool use_gray;
      bool use_rgb;
      bool use_channel_weights;
      bool use_segmentation;

      std::string window_function; //!<  Window function: "hann", "cheb", "kaiser"
      float kaiser_alpha;
      float cheb_attenuation;

      float template_size;
      float gsl_sigma;
      float hog_orientations;
      float hog_clip;
      float padding;
      float filter_lr;
      float weights_lr;
      int num_hog_channels_used;
      int admm_iterations;
      int histogram_bins;
      float histogram_lr;
      int background_ratio;
      int number_of_scales;
      float scale_sigma_factor;
      float scale_model_max_area;
      float scale_lr;
      float scale_step;

      float psr_threshold; //!< we lost the target, if the psr is lower than this.
    };

    // TrackerCSRTImpl(const Params &parameters = Params());

    bool init(cv::Mat &image, const cv::Rect2d &boundingBox);
    bool update(const cv::Mat &image_, cv::Rect2d &boundingBox, const cv::Mat &homoMat);

    void read(const cv::FileNode &fn); // CV_OVERRIDE;
    void write(cv::FileStorage &fs);   // const CV_OVERRIDE;
    void setParams(Params _params)
    {
      params = _params;
    }

  protected:
    Params params;
    cv::Mat framePrevious;
    bool initImpl(const cv::Mat &image, const cv::Rect2d &boundingBox);                    // CV_OVERRIDE;
    virtual void setInitialMask(cv::InputArray mask);                                      // CV_OVERRIDE;
    bool updateImpl(const cv::Mat &image, cv::Rect2d &boundingBox, const cv::Mat homoMat); // CV_OVERRIDE;
    void update_csr_filter(const cv::Mat &image, const cv::Mat &my_mask);
    void update_histograms(const cv::Mat &image, const cv::Rect &region);
    void extract_histograms(const cv::Mat &image, cv::Rect region, Histogram &hf, Histogram &hb);
    std::vector<cv::Mat> create_csr_filter(const std::vector<cv::Mat>
                                               img_features,
                                           const cv::Mat Y, const cv::Mat P);
    cv::Mat calculate_response(const cv::Mat &image, const std::vector<cv::Mat> filter);
    cv::Mat get_location_prior(const cv::Rect roi, const cv::Size2f target_size, const cv::Size img_sz);
    cv::Mat segment_region(const cv::Mat &image, const cv::Point2f &object_center,
                           const cv::Size2f &template_size, const cv::Size &target_size, float scale_factor);
    cv::Point2f estimate_new_position(const cv::Mat &image, double &max_val);
    std::vector<cv::Mat> get_features(const cv::Mat &patch, const cv::Size2i &feature_size);

  private:
    bool isInit;
    int counter_check = 0;
    cv::Ptr<TrackerFeatureSet> featureSet;
    cv::Ptr<TrackerSampler> sampler;
    cv::Ptr<TrackerModel> model;

    bool check_mask_area(const cv::Mat &mat, const double obj_area);
    float current_scale_factor;
    cv::Mat window;
    cv::Mat yf;
    cv::Mat diff;
    cv::Rect2f bounding_box;
    std::vector<cv::Mat> csr_filter;
    std::vector<float> filter_weights;
    cv::Size2f original_target_size;
    cv::Size2i image_size;
    cv::Size2f template_size;
    cv::Size2i rescaled_template_size;
    float rescale_ratio;
    cv::Point2f object_center;
    std::vector<cv::Point2f> points_history;
    std::vector<float> max_val_history;
    std::vector<cv::Mat> region_history;
    int HISTORY_COUNT = 15;
    int THRESH_DISTANCE = 5;
    float THRESH_RESPONSE = 0.1;
    std::vector<cv::Point2f> pointsGrid;

    int falseCounter = 0;
    DSST dsst;
    Histogram hist_foreground;
    Histogram hist_background;
    double p_b;
    cv::Mat erode_element;
    cv::Mat filter_mask;
    cv::Mat preset_mask;
    cv::Mat default_mask;
    float default_mask_area;
    int cell_size;
  };

} /* namespace myCSRT */
#endif
