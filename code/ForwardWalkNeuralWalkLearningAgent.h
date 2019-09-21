
#ifndef TEAM_SOCCER3D_CPP_FORWARDWALKNEURALWALKLEARNINGAGENT_H
#define TEAM_SOCCER3D_CPP_FORWARDWALKNEURALWALKLEARNINGAGENT_H


#include "BaseLearningAgent.h"

#include "soccer3d.grpc.pb.h"
#include <queue>
#include "control/MovementSwitchSmoother.h"
using api::Action;
using api::SetupEnvResponse;
using api::SimulationResponse;
using api::State;


class ForwardWalkNeuralWalkLearningAgent: public BaseLearningAgent {
public:
    ForwardWalkNeuralWalkLearningAgent(std::string host = "127.0.0.1", int serverPort = 3100, int monitorPort = 3200, int agentNumber = 1,
    int robotType = 0, std::string teamName = std::string("TeamName"));

    State newEpisode() override;
    SimulationResponse runStep(Action action) override;
    SetupEnvResponse setup() override;

private:
    // Checks whether the episode has finished
    bool episodeOver();

    // Reward signal
    double reward();


    // Environment state
    State state();

    // Runs simulation step in the environment
    void step();
    void stepControl();


    //agent
    Vector3<double> lastAgentTranslation;
    Vector3<double> agentVelocity;
    std::queue<Vector3<double> > lastTranslations;

    //control
    vector<double> desiredJoints;

    int iEpi = 0;
    int nbSteps;
    int episodeNumber;

    representations::NaoJoints previousJoints;
    double previousOrientation;
    double previousHeight;
    double heightDerivative;
    double leftFootCounter, rightFootCounter;
    bool rightFootWasTouchingGround, leftFootWasTouchingGround;
    representations::NaoJoints jointsDerivatives;
    Vector3<double> previousCoM;
    Vector3<double> previousZMP;
    Vector3<double> previousTorsoVel;
    Vector3<double> previousTorsoAccel;
    Vector3<double> previousRightFootForce, previousLeftFootForce;
    Vector3<double> previousLeftFootForceCoord, previousRightFootForceCoord;
    control::MovementSwitchSmoother smoother;
    std::ofstream runStatistics, fallRate;
    double maxVelocity;

};


#endif
