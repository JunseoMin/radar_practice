#include <radartypes.hpp>
#include "radar_utills.hpp"
#include "FeatureBasedPoseTracker.hpp"

pmc::pmc_graph EigenToPMC(const Eigen::MatrixXd& G){
  vector<long long> vs;
  vector<int>       es;

  vs.push_back(0);

  int tmp = 0;

  for (int vertex = 0; vertex < G.rows(); vertex++)
  {
    int n_edges =  G.row(vertex).sum();
    tmp += n_edges;
    vs.push_back(tmp);
    for (int edge_idx = 0; edge_idx < G.cols(); edge_idx ++){
      if (G(vertex,edge_idx) == 1)
      {
        es.push_back(edge_idx);
      }
    }
  }
  
  // std::cout << "vs: " << std::endl;
  // for (int v : vs) cout << v << " ";
  // std::cout << std::endl;
  // std::cout << "es: " << std::endl;
  // for (int e : es) cout << e << " ";
  // std::cout << std::endl;
  std::cout << "num vertexis: " << vs.size() << "\n";
  pmc::pmc_graph p_G(vs,es);
  return p_G;
}

FeatureBasedPoseTracker::FeatureBasedPoseTracker(const double& optim_threshold){
  // this->_orb                   = cv::ORB::create(1000,); 
  this -> _orb                    = cv::ORB::create(500, 1.4f, 8, 40, 0, 2, 
                                           cv::ORB::HARRIS_SCORE, 50, 30);
  this -> _key_id                 = 0;
  this -> _matcher                = cv::BFMatcher(cv::NORM_HAMMING, true);
  this -> _optim_treshold         = optim_threshold;  // pixel distance
  //---------- pmc initialization param (same as from ORORA pmc initializer)


  std::cout << "[Info] Solver initialized!" << std::endl;
}

void FeatureBasedPoseTracker::setCurrentFrame(cv::Mat& current_frame){
  this->_current_frame = current_frame;
  std::cout << "original size: " << current_frame.rows << " x " <<current_frame.cols << std::endl;
  
  // post process1: cubic interpolation and change size to 640x640 - seems doesn't have much impact..
  cv::resize(_current_frame, _current_frame, cv::Size(640, 640), 0, 0, cv::INTER_LANCZOS4); 
  
  // morphology calc
  // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  // cv::dilate(_current_frame, _current_frame, kernel);
  // cv::morphologyEx(_current_frame, _current_frame, cv::MORPH_CLOSE, kernel);

  std::cout << "[Info] frame set! "<< std::endl;
}

Eigen::Matrix3d FeatureBasedPoseTracker::getPose(){
  return _pose;
}

/**
 * @brief solves SE(2) position matrix
 */
void FeatureBasedPoseTracker::solve(){
  // assert(_current_frame.size() > 0);

  _keypoints.clear();
  _descriptors = cv::Mat();
  
  // 1. get orb feature from frame

  // std::cout << "input channel: " << _current_frame.channels() << std::endl;
  if (_current_frame.empty()) {
    throw std::logic_error("[Error] input frame empty!");
  }
  if (_current_frame.depth() != CV_8UC1) {
    std::cout << "[Info] Converting depth to CV_8UC1 type" << std::endl;
    _current_frame.convertTo(_current_frame, CV_8UC1, 255.0);
  }

  // cv::imshow("origin frame", _current_frame);

  _orb->detectAndCompute(_current_frame, cv::Mat(), _keypoints, _descriptors);
  std::cout << "[Info] orb extracted" << std::endl;
  std::cout << "[Info] num keypoints: " << _keypoints.size() << std::endl;

  // debug... feature doesn't extracted with direct input need pre-processing or change ORB param  - Nope!
  // cv::Mat img_keypoints;
  // cv::drawKeypoints(_current_frame, _keypoints, img_keypoints, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
  // cv::imshow("ORB Keypoints", img_keypoints);
  // cv::waitKey(0.1);

  if(_key_id == 0){ // case initialize
    std::cout << "[Info] Frame initialized!!" << std::endl;
    _pose = Eigen::Matrix3d::Identity();
    std::cout << "initial pose: \n"
              << _pose << std::endl;
    _addKeyframe();
    return;
  }

  // 2. find closest key frame
  _getMatchedKeyframe();
  assert(_best_match_id != -1);
  std::cout << "keyframe id: " << _best_match_id << "\n";

  // 3. solve pose optimization (graph optimization)
  _findGoodMatch();
  _findMaxClique();
  _guessInitialPose();
  _optimizePose();
  _visualizeInliers();

  //4. add keyframe if needed
  if(_isRequireKeyframe()) _addKeyframe();

  _current_frame = cv::Mat(); // clear current frame
  _inliers.clear();
  _matches.clear();
  _p_ks.clear();
  _p_ts.clear();


}

void FeatureBasedPoseTracker::_addKeyframe(){
  this->_key_frames.emplace_back(Keyframe(_key_id, _current_frame, _keypoints, _descriptors, _pose));
  _key_id ++;
  std::cout << "current key frame size: " << _key_frames.size() << std::endl;
  std::cout << "!!!!! Add Keyframe !!!!!!" << _key_frames.size() << std::endl;
  std::cout << "!!!!! Add Keyframe !!!!!!" << _key_frames.size() << std::endl;
  std::cout << "!!!!! Add Keyframe !!!!!!" << _key_frames.size() << std::endl;
  std::cout << "!!!!! Add Keyframe !!!!!!" << _key_frames.size() << std::endl;
}

void FeatureBasedPoseTracker::_getMatchedKeyframe(){
  // find most closest Keyframe from Keyframe vector
  _best_match_id = -1;
  int cnt_best_match = 0;

  for(auto& keyframe: _key_frames){
    std::vector<cv::DMatch> matches;
    _matcher.match(_descriptors, keyframe.descriptors, matches);  // tlqkf durltj qkRiTsp zz

    if (matches.size() > cnt_best_match)
    {
      cnt_best_match = matches.size();
      _best_match_id = keyframe.id;
      _matches = matches;
    }
  }
}

void FeatureBasedPoseTracker::_findGoodMatch(){
  int num_matches = _matches.size();

  _G = Eigen::MatrixXd::Zero(num_matches, num_matches);
  // int n_good = 0;
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < num_matches; i++)
  {
    for (size_t j = i + 1; j < num_matches; j++)
    {
      cv::Point2d p_ik = _key_frames[_best_match_id].keypoints[_matches[i].trainIdx].pt;  // keyframe featurepoint i
      cv::Point2d p_it = _keypoints[_matches[i].queryIdx].pt;  // query featurepoint i

      cv::Point2d p_jk = _key_frames[_best_match_id].keypoints[_matches[j].trainIdx].pt;  // keyframe featurepoint j
      cv::Point2d p_jt = _keypoints[_matches[j].queryIdx].pt;  // query featurepoint j

      double dist_i = std::sqrt(std::pow(p_ik.x - p_it.x, 2) + std::pow(p_ik.y - p_it.y, 2));
      double dist_j = std::sqrt(std::pow(p_jk.x - p_jt.x, 2) + std::pow(p_jk.y - p_jt.y, 2));
      
      // std::cout << std::abs(dist_i - dist_j) << "\n";
      // std::cout << (std::abs(dist_i - dist_j) > _optim_treshold) ? "skip \n" : "add\n";
      if(std::abs(dist_i - dist_j) < _optim_treshold){
        _G(i,j) = 1;
        _G(j,i) = 1; // Symmetric
        // std::cout << "std::abs(dist_i - dist_j): " << std::endl;
        // std::cout << std::abs(dist_i - dist_j) << std::endl;
        // n_good ++;
      }
    }
  }
  // std::cout << "[Debug] original matches size: " << _matches.size() << std::endl 
            // << "[Info] number of good matches: " << _G.sum() / 2 << std::endl;
            // << "[Info] number of good matches: " << n_good << std::endl;
            // << _G << std::endl;


  // assert(_matches.size() > n_good); // 
}

void FeatureBasedPoseTracker::_findMaxClique(){
  // find maxClique
  pmc::input in;
  in.algorithm           = 0;
  in.threads             = 12; // for OMP_NUM_THREADS 12
  in.experiment          = 0;
  in.lb                  = 0;
  in.ub                  = 0;
  in.param_ub            = 0;
  in.adj_limit           = 20000;
  in.time_limit          = 100;
  in.remove_time         = 4;
  in.graph_stats         = false;
  in.verbose             = false;
  in.help                = false;
  in.MCE                 = false;
  in.decreasing_order    = false;
  in.heu_strat           = "kcore";
  in.vertex_search_order = "deg";

  pmc::pmc_graph p_G = EigenToPMC(_G);

  p_G.compute_cores();
  auto max_core = p_G.get_max_core();
  std::cout << "Max core number: " << max_core << "\n";

  if(in.ub == 0){
    in.ub = max_core + 1;
  }

  if (in.lb == 0) { // skip if given as input
    pmc::pmc_heu maxclique(p_G, in);
    in.lb = maxclique.search(p_G, _inliers);
  }

  if (in.lb == in.ub) {
    std::cout << "lb == ub \n";
    return;
  }

  // std::cout<< "find segfault" << std::endl;
  
  std::cout << "exect solver called! \n";
  if(p_G.num_vertices() < in.adj_limit){ // This contains segfault (pmc maxclique solve)
    p_G.create_adj();
    pmc::pmcx_maxclique finder(p_G, in);
    finder.search_dense(p_G, _inliers);
  }
  else{
    pmc::pmcx_maxclique finder(p_G, in);
    finder.search(p_G, _inliers);    
  }
  // std::cout << "inlier index(?):\n";
  // for(auto& inlier : _inliers){
  //   std::cout << inlier << ",";
  // }
  // std::cout << "\n";
  std::cout << "[Info] number of inliers: " << _inliers.size() << std::endl;
  // upper lines are followed by TEASER ++ pipeline
  // olny thing have to do is optimize pose and get C_t
}

// void FeatureBasedPoseTracker::_findMaxClique(){
//   //muuuuuuuuuuuuuuuuch faster but performance sucks check idx.
//   pmc::pmc_graph p_G = EigenToPMC(_G);

//   p_G.compute_cores();
//   auto max_core = p_G.get_max_core();

//   // find maxClique
//   // remove all nodes with core number less than max core number
//   // k_cores is a vector saving the core number of each vertex
//   auto     k_cores = p_G.get_kcores();
//   for (int i       = 1; i < k_cores->size(); ++i) {
//     // Note: k_core has size equals to num vertices + 1
//     if ((*k_cores)[i] >= max_core) {
//       _inliers.push_back(i - 1);  // seems awkward... this contains index of _matcher
//     }
//   }

//   // std::cout << "!inliers: \n";
//   // for(auto& inlier : _inliers){
//   //   std::cout << inlier << ",";
//   // }
// }

void FeatureBasedPoseTracker::_guessInitialPose(){
  // ORORA like algorithms(e.g. TEASER++ etc...) doesn't need like this initial pose estimation,, But..! let's try!
  // This module guess initial pose by SVD
  std::vector<Eigen::Vector2d> p_ks;
  std::vector<Eigen::Vector2d> p_ts;

  std::cout << "num inliers: " << _inliers.size() << "\n";

  for(auto& i : _inliers){
    Eigen::Vector2d p_t(_keypoints[_matches[i].queryIdx].pt.x,
                        _keypoints[_matches[i].queryIdx].pt.y);
    Eigen::Vector2d p_k(_key_frames[_best_match_id].keypoints[_matches[i].trainIdx].pt.x,
                        _key_frames[_best_match_id].keypoints[_matches[i].trainIdx].pt.y);
    p_ts.emplace_back(p_t);
    p_ks.emplace_back(p_k);
  }

  _p_ts = p_ts;
  _p_ks = p_ks;

  assert(p_ts.size() == p_ks.size());
  int N = p_ts.size(); // number of matched points

  Eigen::Vector2d centroid_k = Eigen::Vector2d::Zero();
  Eigen::Vector2d centroid_t = Eigen::Vector2d::Zero();

  for (int i = 0; i < N; i++){
    centroid_k += p_ks[i];
    centroid_t += p_ts[i];
  }

  centroid_k /= N;
  centroid_t /= N;
  
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 2);

  for (int i = 0; i < N; i++){
    Eigen::Vector2d pk = p_ks[i] - centroid_k;
    Eigen::Vector2d pt = p_ts[i] - centroid_t;
    H += pk * pt.transpose();
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix2d U = svd.matrixU();
  Eigen::Matrix2d V = svd.matrixV();

  Eigen::Matrix2d R = U * V.transpose();
  if (R.determinant() < 0) {
    V.col(1) *= -1;
    R = V * U.transpose();
  }

  Eigen::Vector2d t = centroid_t - R * centroid_k;
  _initial_pose = Eigen::Matrix3d::Identity();
  _initial_pose.block<2,2>(0,0) = R;
  _initial_pose.block<2,1>(0,2) = t;
  std::cout << "initial pose: \n" << _initial_pose << std::endl;
}

void FeatureBasedPoseTracker::_optimizePose(){
  std::cout << "[Info] optimize start" << std::endl;
  double pose[3] = {_initial_pose(0,2), _initial_pose(1,2), atan2(_initial_pose(1,0),_initial_pose(0,0))};  // translation to keyframe
  // double pose[3] = {1, 1, atan2(1,1)};  // translation to keyframe

  ceres::Problem               problem; //equation(4)
  std::vector<Eigen::Vector2d> P_w; // keyframe points in word coordinate
  // std::cout << _p_ks.size() << "?!?!?!?!?!?!?!?!??" << std::endl;

  for (const auto& p_k : _p_ks){
    Eigen::Vector3d pk_h(p_k(0),p_k(1),1);
    Eigen::Vector3d pw_h = _key_frames[_best_match_id].pose * pk_h;

    P_w.emplace_back(pw_h.head<2>());
  }

  for (size_t i = 0; i < P_w.size(); i++)
  {
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<PoseCostFunctor, 2, 3>(
      new PoseCostFunctor(P_w[i], _p_ts[i])
      );
    problem.AddResidualBlock(cost_function, nullptr, pose);
  }

  assert(problem.NumResidualBlocks() != 0);
  // std::cout << "Number of residual blocks: " << problem.NumResidualBlocks() << std::endl;

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // std::cout << summary.BriefReport() << std::endl;

  _pose << cos(pose[2]), -sin(pose[2]), pose[0],
           sin(pose[2]),  cos(pose[2]), pose[1],
           0,             0,             1;
}

bool FeatureBasedPoseTracker::_isRequireKeyframe(){
  Eigen::Vector2d t_k = _key_frames[_best_match_id].pose.block<2,1>(0,2);
  Eigen::Vector2d t_f = _pose.block<2,1>(0,2);

  Eigen::Matrix2d r_k = _key_frames[_best_match_id].pose.block<2,2>(0,0);
  Eigen::Matrix2d r_f = _pose.block<2,2>(0,0);
  
  double dist = (t_k - t_f).norm();
  Eigen::Matrix2d diff = r_k.transpose() * r_f;

  double angle_diff = std::atan2(diff(1, 0), diff(0, 0));

  if(std::abs(dist) > 8) return true;
  // if(std::abs(angle_diff) > 90) return true;

  return false;
}

void FeatureBasedPoseTracker::_visualizeInliers() {
  // cv::Mat frame_visual;
  cv::Mat frame_visual_comp;
  if (_current_frame.channels() == 1) {
    // cv::cvtColor(_current_frame, frame_visual, cv::COLOR_GRAY2BGR);
    cv::cvtColor(_current_frame, frame_visual_comp, cv::COLOR_GRAY2BGR);
  } else {
    // frame_visual = _current_frame.clone();
    throw std::logic_error("Need 8UC1 type");
  }
  cv::Mat keyframe_visual;
  if (_key_frames[_best_match_id].image.channels() == 1) {
    cv::cvtColor(_key_frames[_best_match_id].image, keyframe_visual, cv::COLOR_GRAY2BGR);
  } else {
    keyframe_visual = _key_frames[_best_match_id].image.clone();
  }
  #pragma omp parallel
  for (const auto& idx : _inliers) {
    cv::Point2d pt_t = _keypoints[_matches[idx].queryIdx].pt;
    cv::Point2d pt_k = _key_frames[_best_match_id].keypoints[_matches[idx].trainIdx].pt;
    // cv::circle(frame_visual, pt_t, 5, cv::Scalar(0, 255, 0), -1);
    #pragma omp critical
    {
      cv::circle(keyframe_visual, pt_t, 2, cv::Scalar(0, 255, 0), -1);
      // cv::circle(keyframe_visual, pt_k, 5, cv::Scalar(0, 0, 255), -1);
      cv::circle(keyframe_visual, pt_k, 2, cv::Scalar(0, 0, 255), -1);
      cv::line(keyframe_visual, pt_t, pt_k, cv::Scalar(255, 0, 0), 1);
    }
  }
  cv::imshow("frame visual comp", keyframe_visual);
}


//surf cv implementation