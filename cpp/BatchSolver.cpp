#include <iostream>
#include "BatchSolver.hpp"
#include "P2P3ErrorEval.hpp"
#include "LidarImuCalibPriorEval.hpp"
#include "mcransac.hpp"


// Set the Qc inverse matrix with the diagonal of Qc
void BatchSolver::setQcInv(const np::ndarray& Qc_diag) {
    Eigen::Matrix<double, 6, 1> temp = numpyToEigen2D(Qc_diag);
    Qc_inv_.setZero();
    Qc_inv_.diagonal() = 1.0/temp.array();
}

// add new state variable at specified time
void BatchSolver::addNewState(double time) {
    Eigen::Matrix<double, 6, 1> zero_vel;
    zero_vel.setZero();

    // states vector
    TrajStateVar temp;
    temp.time = steam::Time(time);
    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(lgmath::se3::Transformation()));
    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(zero_vel));
    states_.push_back(temp);

    // steam trajectory
    TrajStateVar& state = states_.back();
    steam::se3::TransformStateEvaluator::Ptr tse = steam::se3::TransformStateEvaluator::MakeShared(state.pose);
    traj_.add(state.time, tse, state.velocity);
}

// input point matrices are Nx2
void BatchSolver::addFramePair(const np::ndarray& p2, const np::ndarray& p1,
    const np::ndarray& t2, const np::ndarray& t1, const int64_t& earliest_time, const int64_t& latest_time) {

    // NOTE: I think we're possibly skipping a state for the 2nd radar scan, but shouldn't matter too much
    // add new state variable if first
    if (states_.empty()) {
        time_ref_ = earliest_time;
        addNewState(0.0);
    }

    // add new state variable at latest time
    double delta_t = double(latest_time - time_ref_) / 1.0e6;
    addNewState(delta_t);
    std::cout << "Added new state at: " << delta_t << " seconds" << std::endl;

    // ransac
    std::vector<int> inliers;
    if (use_ransac) {
        srand(time_ref_ / 1.0e6);  // fix random seed for repeatability
        Eigen::VectorXd motion_vec = Eigen::VectorXd::Zero(6);
        Eigen::MatrixXd T;

        Ransac ransac(p1, p2);
        ransac.computeModel();
        ransac.getTransform(T);
        ransac.getInliers(T, inliers);

    } else {
        for (uint j = 0; j < p1.shape(0); ++j) {
            inliers.push_back(j);
        }
    }

    // loop through each match pair
    steam::GemanMcClureLossFunc::Ptr sharedLossFuncGM(new steam::GemanMcClureLossFunc(1.0));
    for (uint k = 0; k < inliers.size(); ++k) {
        uint j = inliers[k];    // index of inlier
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();    // set weight to identity (TODO: allow scalar tuning)
        steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::INFORMATION));

        // get measurement
        Eigen::Vector4d read;
        read << double(p::extract<float>(p2[j][0])), double(p::extract<float>(p2[j][1])), 0.0, 1.0;

        Eigen::Vector4d ref;
        ref << double(p::extract<float>(p1[j][0])), double(p::extract<float>(p1[j][1])), 0.0, 1.0;

        // get relative pose expression
        int64_t ta_ = int64_t(p::extract<int64_t>(t1[j])) - time_ref_;
        int64_t tb_ = int64_t(p::extract<int64_t>(t2[j])) - time_ref_;
        double ta = double(ta_) / 1.0e6;
        double tb = double(tb_) / 1.0e6;
        // steam::se3::TransformStateEvaluator::Ptr Tsv = steam::se3::TransformStateEvaluator::MakeShared(T_s_v_);

        steam::se3::TransformStateEvaluator::Ptr Tfl = steam::se3::TransformStateEvaluator::MakeShared(T_fl_);
        steam::se3::TransformEvaluator::Ptr Trl = steam::se3::compose(T_rf_, Tfl);
        steam::se3::TransformEvaluator::Ptr Trv = steam::se3::compose(Trl, T_lv_);

        steam::se3::TransformEvaluator::ConstPtr Ta0 = traj_.getInterpPoseEval(steam::Time(ta));
        steam::se3::TransformEvaluator::ConstPtr Tb0 = traj_.getInterpPoseEval(steam::Time(tb));
        steam::se3::TransformEvaluator::Ptr T_eval_ptr = steam::se3::composeInverse(
            steam::se3::compose(Trv, Tb0),
            steam::se3::compose(Trv, Ta0));  // Tba = Tb0 * inv(Ta0)

        // add cost
        steam::P2P3ErrorEval::Ptr error(new steam::P2P3ErrorEval(ref, read, T_eval_ptr));
        steam::WeightedLeastSqCostTerm<3, 6>::Ptr cost(
            new steam::WeightedLeastSqCostTerm<3, 6>(error, sharedNoiseModel, sharedLossFuncGM));
        cost_terms_->add(cost);
    }
}

// Run optimization
void BatchSolver::optimize() {

    // lock first pose
    states_[0].pose->setLock(true);
    T_fl_->setLock(true);  // temporary

    // additional cost terms
    steam::ParallelizedCostTermCollection::Ptr costs(new steam::ParallelizedCostTermCollection());

    // prior on extrinsic
    steam::L2LossFunc::Ptr sharedLossFuncL2(new steam::L2LossFunc());
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    R(0, 0) = 0.01*0.01;    // variance of z-offset
    R(1, 1) = 0.001*0.001;  // variance of roll-offset
    R(2, 2) = 0.001*0.001;  // variance of elevation-offset
    steam::BaseNoiseModel<3>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<3>(R, steam::COVARIANCE));
    steam::se3::TransformStateEvaluator::Ptr Tfl = steam::se3::TransformStateEvaluator::MakeShared(T_fl_);
    steam::LidarImuCalibPriorEval::Ptr error(new steam::LidarImuCalibPriorEval(z_offset_, Tfl));
    steam::WeightedLeastSqCostTerm<3, 6>::Ptr cost(
        new steam::WeightedLeastSqCostTerm<3, 6>(error, sharedNoiseModel, sharedLossFuncL2));
    costs->add(cost);

    // WNOA
    std::cout << "Getting WNOA prior terms..." << std::endl;
    traj_.appendPriorCostTerms(costs);

    steam::OptimizationProblem problem;

    // Add state variables
    std::cout << "Adding state variables..." << std::endl;
    for (uint i = 0; i < states_.size(); ++i) {
        const TrajStateVar& state = states_.at(i);
        problem.addStateVariable(state.pose);
        problem.addStateVariable(state.velocity);
    }
    problem.addStateVariable(T_fl_);   // extrinsic

    std::cout << "Adding cost terms..." << std::endl;
    problem.addCostTerm(costs);
    problem.addCostTerm(cost_terms_);
    SolverType::Params params;
    params.verbose = true;
    solver_ = SolverBasePtr(new SolverType(&problem, params));

    std::cout << "Optimizing..." << std::endl;
    solver_->optimize();
    std::cout << "Complete." << std::endl;
}

void BatchSolver::getPoses(np::ndarray& poses) {
    for (uint i = 0; i < states_.size(); ++i) {
//        Eigen::Matrix<double, 4, 4> Tsi =
//            T_s_v_->getValue().matrix()*states_[i].pose->getValue().matrix()*T_s_v_->getValue().inverse().matrix();
        Eigen::Matrix<double, 4, 4> Tvi = states_[i].pose->getValue().matrix();
        for (uint r = 0; r < 3; ++r) {
            for (uint c = 0; c < 4; ++c) {
//                poses[i][r][c] = float(Tsi(r, c));
                poses[i][r][c] = float(Tvi(r, c));
            }
        }
    }
}

void BatchSolver::getPath(np::ndarray& path) {
    for (uint i = 0; i < states_.size(); ++i) {
        Eigen::Matrix<double, 3, 1> r_vi_in_i = states_[i].pose->getValue().r_ba_ina();
        for (uint r = 0; r < 3; ++r) {
            path[i][r] = float(r_vi_in_i(r));
        }
    }
}

void BatchSolver::getVelocities(np::ndarray& vels) {
    for (uint i = 0; i < states_.size(); ++i) {
        Eigen::Matrix<double, 6, 1> vel = states_[i].velocity->getValue();
        for (uint r = 0; r < 6; ++r) {
            vels[i][r] = float(vel(r));
        }
    }
}