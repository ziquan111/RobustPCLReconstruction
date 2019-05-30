// Refer to the paper: "Robust Point Cloud Based Reconstruction of Large-Scale Outdoor Scenes" in CVPR 2019
// Link: https://arxiv.org/abs/1905.09634

// This file is an implememntation of the Gaussian-Uniform mixture model as described in Sec. 5 of the paper.
// The Gaussian-Uniform mixture model degenerates to the approach in Choi et al. "Robust Reconstruction of Indoor Scenes" in CVPR 2017.
// This implementation adopts some techniques used by Choi et al.

#include <ceres/ceres.h>
#include <iostream>
#include <sophus/se3.hpp>

#include <vector>
#include <fstream>
#include <Eigen/Core>

#include <iomanip>
#include <algorithm>

#include "local_parameterization_se3.hpp"

#define NUM_ITERATION_EM 1000

#define EPSILON 6                             // upper bound of feature matching errors, used in Eq. (27)
#define M_HAT EPSILON * EPSILON               // mean error, used in Eq. (27)
#define P_HAT 0.9                             // trust level, used in Eq. (27)
#define THETA P_HAT / ( 1.0 - P_HAT) * M_HAT  // THETA is solved using Eq. (27)

#define THESHOLD_TO_BE_AN_INLIER_LOOP 0.8 * P_HAT    // theshold to prune outlier loop closures, used to generate results in Tab. 1 and 2

typedef Eigen::Matrix< double, 6, 6, Eigen::RowMajor > InformationMatrix;
typedef Eigen::Matrix< double, 6, 1> Vector6d;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct FramedTransformation {
  int id1_;
  int id2_;
  int frame_;
  Eigen::Matrix4d transformation_;      // pose in matrix form
  Sophus::SE3d transformation_se3_;     // pose in se3 form

  FramedTransformation( int id1, int id2, int f, Eigen::Matrix4d t )
    : id1_( id1 ), id2_( id2 ), frame_( f ), transformation_( t ) 
  {
    Eigen::Quaterniond q;
    q = t.block<3,3>(0,0);
    transformation_se3_ = Sophus::SE3d(q, Sophus::SE3d::Point(t(0,3), t(1,3), t(2,3)));
  }

  FramedTransformation(int id1, int id2, int f, Sophus::SE3d t)
    : id1_( id1 ), id2_( id2 ), frame_( f ), transformation_se3_( t ) 
    {
      transformation_ = t.matrix();
    }
};

struct PCLTrajectory {
  std::vector< FramedTransformation > data_;
  int index_;

  void LoadFromFile(const char* filename ) {
    data_.clear();
    index_ = 0;
    int id1, id2, frame;
    Eigen::Matrix4d trans;
    FILE * f = fopen( filename, "r" );
    if ( f != NULL ) {
      char buffer[1024];
      while ( fgets( buffer, 1024, f ) != NULL ) {
        if ( strlen( buffer ) > 0 && buffer[ 0 ] != '#' ) {
          sscanf( buffer, "%d %d %d", &id1, &id2, &frame);
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf", &trans(0,0), &trans(0,1), &trans(0,2), &trans(0,3) );
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf", &trans(1,0), &trans(1,1), &trans(1,2), &trans(1,3) );
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf", &trans(2,0), &trans(2,1), &trans(2,2), &trans(2,3) );
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf", &trans(3,0), &trans(3,1), &trans(3,2), &trans(3,3) );
          data_.push_back( FramedTransformation( id1, id2, frame, trans ) );
        }
      }
      fclose( f );
    }
  }

  void SaveToFile(const char* filename ) {
    FILE * f = fopen( filename, "w" );
    for ( int i = 0; i < ( int )data_.size(); i++ ) {
      Sophus::SE3d trans_se3 = data_[ i ].transformation_se3_;
      Eigen::Matrix4d trans = trans_se3.matrix();
      fprintf( f, "%d\t%d\t%d\n", data_[ i ].id1_, data_[ i ].id2_, data_[ i ].frame_ );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(0,0), trans(0,1), trans(0,2), trans(0,3) );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(1,0), trans(1,1), trans(1,2), trans(1,3) );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(2,0), trans(2,1), trans(2,2), trans(2,3) );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(3,0), trans(3,1), trans(3,2), trans(3,3) );

    }
    fclose( f );
  }

};

struct FramedInformation {
  int id1_;
  int id2_;
  int frame_;
  InformationMatrix information_;

  FramedInformation( int id1, int id2, int f, InformationMatrix t )
    : id1_( id1 ), id2_( id2 ), frame_( f ), information_( t ) 
  {}
};

struct PCLInformation {
  std::vector< FramedInformation > data_;

  void LoadFromFile(const char*  filename ) {
    data_.clear();
    int id1, id2, frame;
    InformationMatrix info;
    FILE * f = fopen( filename, "r" );
    if ( f != NULL ) {
      char buffer[1024];
      while ( fgets( buffer, 1024, f ) != NULL ) {
        if ( strlen( buffer ) > 0 && buffer[ 0 ] != '#' ) {
          sscanf( buffer, "%d %d %d", &id1, &id2, &frame);
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf %lf %lf", &info(0,0), &info(0,1), &info(0,2), &info(0,3), &info(0,4), &info(0,5) );
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf %lf %lf", &info(1,0), &info(1,1), &info(1,2), &info(1,3), &info(1,4), &info(1,5) );
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf %lf %lf", &info(2,0), &info(2,1), &info(2,2), &info(2,3), &info(2,4), &info(2,5) );
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf %lf %lf", &info(3,0), &info(3,1), &info(3,2), &info(3,3), &info(3,4), &info(3,5) );
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf %lf %lf", &info(4,0), &info(4,1), &info(4,2), &info(4,3), &info(4,4), &info(4,5) );
          fgets( buffer, 1024, f );
          sscanf( buffer, "%lf %lf %lf %lf %lf %lf", &info(5,0), &info(5,1), &info(5,2), &info(5,3), &info(5,4), &info(5,5) );
          data_.push_back( FramedInformation( id1, id2, frame, info ) );
        }
      }
      fclose( f );
    }
  }
};

struct GaussianFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GaussianFunctor(Sophus::SE3d trans_next_to_current, InformationMatrix info, int id1, int id2) 
  : trans_next_to_current_(trans_next_to_current), info_(info), id1_(id1), id2_(id2) {}

  template <typename T> 
  bool operator()(T const* const sT_current, T const* const sT_next, 
                  T* residual) const {

    Eigen::Map<Sophus::SE3<T> const> const T_current(sT_current);
    Eigen::Map<Sophus::SE3<T> const> const T_next(sT_next);
    
    Sophus::SE3<T> xi_7T = trans_next_to_current_ 
                          * T_next.inverse() 
                          * T_current;

    // xi: local parameterization as in Choi et al.
    Eigen::Matrix<T,6,1> xi;
    xi << xi_7T.data()[4], xi_7T.data()[5], xi_7T.data()[6], xi_7T.data()[0], xi_7T.data()[1], xi_7T.data()[2];

    Eigen::Matrix<T,1,1> sq_error = xi.transpose() * info_ * xi;
    residual[0] = T( sqrt(sq_error(0,0)) );
    
    return true;
  }

private:
  const Sophus::SE3d trans_next_to_current_;
  const InformationMatrix info_;
  const int id1_, id2_;
};

class RobustPCLReconstruction_GaussianUniform {
public:

  // Odometry contraints
  PCLTrajectory odometry_log_;
  PCLInformation odometry_info_;

  // Loop closure contraints
  PCLTrajectory loop_log_;
  PCLInformation loop_info_;

  // Fragment poses
  PCLTrajectory fragment_poses_;
  PCLTrajectory last_fragment_poses_;   // used to check for EM convergence

  // P_ij table (Eq. 25)
  std::vector< std::vector<double> > P_;
  double average_P_;
  double last_average_P_;   // used to check for EM convergence

  RobustPCLReconstruction_GaussianUniform()
  {
    average_P_ = -1.;
    last_average_P_ = -1.;
  }

  ~RobustPCLReconstruction_GaussianUniform() {}

  void LoadOdometryLog(const char* filename) {
    odometry_log_.LoadFromFile(filename);
  }

  void LoadOdometryInfo(const char* filename) {
    odometry_info_.LoadFromFile(filename);
  }

  void LoadLoopLog(const char* filename) {
    loop_log_.LoadFromFile(filename);
  }

  void LoadLoopInfo(const char* filename) {
    loop_info_.LoadFromFile(filename);
  }

  void InitPosesFromFile(const char* filename) {
    fragment_poses_.LoadFromFile(filename);

    // Initialize P_ij table
    P_.resize(NumPoses());
    for (int i = 0; i < NumPoses(); i++) {
      P_[i].resize(NumPoses());
    }
  }

  // void InitPoses() {
  //   fragment_poses_.data_.clear();
  //   fragment_poses_.index_ = 0;
  //   Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  //   fragment_poses_.data_.push_back( FramedTransformation( 0, 0, 1, pose ) );

  //   for( std::vector< FramedTransformation >::iterator it = odometry_log_.data_.begin();
  //     it != odometry_log_.data_.end(); it ++)
  //   {      
  //     pose = pose * it->transformation_; 
  //     fragment_poses_.data_.push_back( FramedTransformation( it->id2_, it->id2_, it->id2_ + 1, pose ) );
  //   }
  //   std::cout << "total data size: " << odometry_log_.data_.size() << std::endl;

  //   P_.resize(NumPoses());
  //   for (int i = 0; i < NumPoses(); i++) {
  //     P_[i].resize(NumPoses());
  //   }
  // }


  void SavePoses(const char* filename) {
    fragment_poses_.SaveToFile(filename);
  }

  void SaveLinks(const char* filename) {
    FILE * f = fopen( filename, "w" );

    for ( int i = 0; i < NumOdometryConstraints(); i++ ) {
      Sophus::SE3d trans_se3 = odometry_log_.data_[ i ].transformation_se3_;
      Eigen::Matrix4d trans = trans_se3.matrix();
      fprintf( f, "%d\t%d\t%d\n", odometry_log_.data_[ i ].id1_, odometry_log_.data_[ i ].id2_, odometry_log_.data_[ i ].frame_ );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(0,0), trans(0,1), trans(0,2), trans(0,3) );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(1,0), trans(1,1), trans(1,2), trans(1,3) );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(2,0), trans(2,1), trans(2,2), trans(2,3) );
      fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(3,0), trans(3,1), trans(3,2), trans(3,3) );
    }

    for (int i = 0; i < NumLoopClosureConstraints(); i++ ) {
      Sophus::SE3d trans_se3 = loop_log_.data_[ i ].transformation_se3_;
      int id1 = loop_log_.data_[ i ].id1_;
      int id2 = loop_log_.data_[ i ].id2_;
      assert(id1 < id2);
      if (id1 != id2 - 1 && P_[id1][id2] > THESHOLD_TO_BE_AN_INLIER_LOOP) {
        Eigen::Matrix4d trans = trans_se3.matrix();
        fprintf( f, "%d\t%d\t%d\n", loop_log_.data_[ i ].id1_, loop_log_.data_[ i ].id2_, loop_log_.data_[ i ].frame_ );
        fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(0,0), trans(0,1), trans(0,2), trans(0,3) );
        fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(1,0), trans(1,1), trans(1,2), trans(1,3) );
        fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(2,0), trans(2,1), trans(2,2), trans(2,3) );
        fprintf( f, "%.8f %.8f %.8f %.8f\n", trans(3,0), trans(3,1), trans(3,2), trans(3,3) );
      }
    }
    fclose( f );
  }

  void Expectation() {
    std::cout << std::endl << "This is the beginning of EM iteration " << fragment_poses_.index_ + 1 << std::endl
              << ">>>> Expectation Step <<<<" << std::endl;

    double sum_P = 0;   // used to compute average_P_ later
    std::vector<int> P_ij_counter(20);
    for (int ii = 0; ii < 20; ii ++)
      P_ij_counter[ii] = 0;
    
    std::cout << std::setw(5) << "id1"
              << std::setw(5) << "id2"
              << std::setw(10) << "#matches"
              << std::setw(15) << "xiLxi"
              << std::setw(15) << "B_ij"
              << std::setw(15) << "B_ij^2"
              << std::setw(15) << "expterm"
              << std::setw(5) << " --> "
              << std::setw(15) << "P_ij"
              << std::endl;

    // Compute P_ij table
    for (int i = 0; i < NumLoopClosureConstraints(); i ++) {

      FramedTransformation trans = loop_log_.data_[i];
      FramedInformation info = loop_info_.data_[i];

      assert(trans.id1_ == info.id1_ && trans.id2_ == info.id2_ && trans.id1_ < trans.id2_);
      int id1 = info.id1_;
      int id2 = info.id2_;

      // Compute the matrix form of xi
      Sophus::SE3d M_xi = trans.transformation_se3_
                          * fragment_poses_.data_[id2].transformation_se3_.inverse() 
                          * fragment_poses_.data_[id1].transformation_se3_;

      // Extract the vector form of xi
      Vector6d xi;
      xi << M_xi.data()[4], M_xi.data()[5], M_xi.data()[6], M_xi.data()[0], M_xi.data()[1], M_xi.data()[2];

      // Compute B_ij (Eq. 23)
      // This approximiation technique is adopted from Choi et al.
      double xiLxi = xi.transpose() * info.information_ * xi; 
      double B_ij = xiLxi / info.information_(0,0);

      // Compute P_ij (Eq. 25)
      // The exponential-of-log trick is used to avoid the numerical issues when B_ij is too large.
      double expterm = exp(2 * log(B_ij) - log (THETA));
      P_[id1][id2] = 1./ (1. + expterm);

      // Accumulate sum_P
      sum_P += P_[id1][id2];

      // Visualize intermediate steps
      std::cout << std::setw(5) << id1
                << std::setw(5) << id2
                << std::setw(10) << info.information_(0,0)
                << std::setw(15) << xiLxi
                << std::setw(15) << B_ij
                << std::setw(15) << B_ij * B_ij
                << std::setw(15) << expterm
                << std::setw(5) << " --> "
                << std::setw(15) << P_[id1][id2]
                << std::endl;

      // Use P_ij_counter to visualize the distribution of P_ij
      for (int ii = 0; ii < 20; ii ++)
        if (P_[id1][id2] > 0.05 * ii && P_[id1][id2] <= 0.05 * (ii+1))
          P_ij_counter[ii] ++;

    }

    // Save
    last_average_P_ = average_P_;
    last_fragment_poses_.index_ = fragment_poses_.index_;
    last_fragment_poses_.data_.clear();
    for (int i = 0; i < NumPoses(); i++)
      last_fragment_poses_.data_.push_back( FramedTransformation( fragment_poses_.data_[i].id1_, 
                                                                  fragment_poses_.data_[i].id2_, 
                                                                  fragment_poses_.data_[i].frame_,
                                                                  fragment_poses_.data_[i].transformation_se3_));
    // Update
    average_P_ = sum_P / NumLoopClosureConstraints();
    fragment_poses_.index_++;

    // Visualize the distribution of P_ij
    std::cout << "P_ij distributed in 20 bins :" << std::endl;
    for (int ii = 0; ii < 20 ; ii++)
      std::cout << "between" 
                << std::setw(5) << 0.05 * ii  
                << std::setw(5) << "and"
                << std::setw(5) << 0.05 * (ii+1)
                << std::setw(15) << P_ij_counter[ii]
                << std::endl;
  }

  void Maximization() {
    std::cout << std::endl << ">>>> Maximization Step <<<<" << std::endl;

    // Build the problem.
    ceres::Problem problem;

    // Specify local update rule for the parameters
    for (std::vector< FramedTransformation >::iterator it = fragment_poses_.data_.begin(); 
         it != fragment_poses_.data_.end(); it++ ) {
      problem.AddParameterBlock(it->transformation_se3_.data(), Sophus::SE3d::num_parameters,
                                new Sophus::test::LocalParameterizationSE3);
    }

    // Create and add cost functions. Derivatives will be evaluated via
    // automatic differentiation
    for (int i = 0; i < NumOdometryConstraints(); i++) {

      FramedTransformation trans = odometry_log_.data_[i];
      FramedInformation info = odometry_info_.data_[i];
      
      assert(trans.id1_ == info.id1_ && trans.id2_ == info.id2_ && trans.id1_ < trans.id2_);
      int id1 = info.id1_;
      int id2 = info.id2_;

      if (info.information_(0,0) <= 1)
        continue;

      ceres::CostFunction* cost_odometry =
          new ceres::AutoDiffCostFunction<GaussianFunctor, 1,
                                          Sophus::SE3d::num_parameters,
                                          Sophus::SE3d::num_parameters>(
              new GaussianFunctor(trans.transformation_se3_, info.information_, id1, id2));
      problem.AddResidualBlock(cost_odometry, NULL, 
                               fragment_poses_.data_[id1].transformation_se3_.data(), 
                               fragment_poses_.data_[id2].transformation_se3_.data());
    }


    for (int i = 0; i < NumLoopClosureConstraints(); i++) {

      FramedTransformation trans = loop_log_.data_[i];
      FramedInformation info = loop_info_.data_[i];
      
      assert(trans.id1_ == info.id1_ && trans.id2_ == info.id2_ && trans.id1_ < trans.id2_);
      int id1 = info.id1_;
      int id2 = info.id2_;

      ceres::CostFunction* cost_loop =
          new ceres::AutoDiffCostFunction<GaussianFunctor, 1,
                                          Sophus::SE3d::num_parameters,
                                          Sophus::SE3d::num_parameters>(
              new GaussianFunctor(trans.transformation_se3_, info.information_, id1, id2));
      problem.AddResidualBlock(cost_loop, new ceres::ScaledLoss(NULL, P_[id1][id2], ceres::TAKE_OWNERSHIP), 
                               fragment_poses_.data_[id1].transformation_se3_.data(), 
                               fragment_poses_.data_[id2].transformation_se3_.data());
    }
    
    // Set solver options
    ceres::Solver::Options options;
    // options.max_num_iterations = 1000;
    options.gradient_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
    options.function_tolerance = 0.01 * Sophus::Constants<double>::epsilon();
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  
    // Solve and report
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    std::cout << "This is the end of EM iteration : " << fragment_poses_.index_ << std::endl;
  }

  int NumPoses() {
    return fragment_poses_.data_.size();
  }

  int NumOdometryConstraints() {
    return odometry_log_.data_.size();
  }

  int NumLoopClosureConstraints() {
    return loop_log_.data_.size();
  }

  bool IsConverged() {
    if (last_average_P_ < 0) {
      return false;         // not started yet
    }

    std::cout << "Checking for the convergence of average_P_ ...\t";
    if (fabs(average_P_ - last_average_P_) > 0.00001) {
      std::cout << "NOT coverged yet, as average_P_ was updated from " << last_average_P_ << " to " << average_P_ << std::endl;
      return false;
    }
    std::cout << "CONVERGED!" << std::endl;


    std::cout << "Checking for the convergence of poses ...\t";
    for (int i = 0; i < NumPoses(); i++)
    {
      Sophus::SE3d last_pose = last_fragment_poses_.data_[i].transformation_se3_;
      Sophus::SE3d current_pose = fragment_poses_.data_[i].transformation_se3_;

      double const mse = (last_pose.inverse() * current_pose).log().squaredNorm();
      bool const converged = mse < 10. * Sophus::Constants<double>::epsilon();

      if (!converged) {
        std::cout << "NOT converged yet." << std::endl;
        return false;
      }
    }
    std::cout << "CONVERGED!" << std::endl;
    return true;
  }
};

int main(int argc, char** argv) {
  if (argc != 8)
  {
    std::cout << "Please provide the following arguments: InputOdometryLog, InputOdometryInfo, InputLoopLog, InputLoopInfo, InputInitPoses, OutputFinalPoses, OutputKeptLoops" << std::endl;
    return 1;
  } 
  RobustPCLReconstruction_GaussianUniform RobustPCLReconstruction_g;

  RobustPCLReconstruction_g.LoadOdometryLog(argv[1]);
  RobustPCLReconstruction_g.LoadOdometryInfo(argv[2]);
  RobustPCLReconstruction_g.LoadLoopLog(argv[3]);
  RobustPCLReconstruction_g.LoadLoopInfo(argv[4]);
  RobustPCLReconstruction_g.InitPosesFromFile(argv[5]);

  bool isConverged = false;
  for (int i = 0 ; i < NUM_ITERATION_EM; i ++) {
    RobustPCLReconstruction_g.Expectation();
    RobustPCLReconstruction_g.Maximization();
    if (RobustPCLReconstruction_g.IsConverged()) {
      std::cout << "Optimization ends since EM is converged" << std::endl << std::endl;
      isConverged = true;
      break;
    }
  }

  if (!isConverged)
    std::cout << "Optimization ends after " << NUM_ITERATION_EM << " EM iterations" << std::endl << std::endl;

  RobustPCLReconstruction_g.SavePoses(argv[6]);
  std::cout << "Final poses are saved to " << argv[6] << std::endl;

  RobustPCLReconstruction_g.SaveLinks(argv[7]);
  std::cout << "Good links are saved to " << argv[7] << std::endl;

  std::cout << "There are " << RobustPCLReconstruction_g.NumPoses() << " poses" << std::endl;
  std::cout << "There are " << RobustPCLReconstruction_g.NumOdometryConstraints() << " odometry constraints" << std::endl;
  std::cout << "There are " << RobustPCLReconstruction_g.NumLoopClosureConstraints() << " loops" << std::endl;

  std::cout << std::endl;
  
  return 0;
}
