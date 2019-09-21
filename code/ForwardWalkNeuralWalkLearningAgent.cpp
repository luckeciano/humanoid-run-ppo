
#include "external/easylogging++.h"
#include "ForwardWalkNeuralWalkLearningAgent.h"
#include <math.h>
#include "core/control/requests/walk/WalkRequest.h"

// RL
const int NUMBER_OF_STATE_DIM = 89;
const int NUMBER_OF_ACTION_DIM = 20;
const int NUM_STEP_SAME_INPUT = 1;
const double STEP_TIME = 0.02;
const int N_STEPS_REWARD = 2;
double EPISODE_LENGTH = 400; 
const Vector3<double> INITIAL_AGENT_POS = Vector3<double>(-14.0, 0.0, 0.0);
const Vector3<double> TARGET_POS = Vector3<double>(14.0, 0.0, 0.0);

ForwardWalkNeuralWalkLearningAgent::ForwardWalkNeuralWalkLearningAgent(std::string host, int serverPort, int monitorPort, int agentNumber,
                                       int robotType, std::string teamName)
        : BaseLearningAgent(host, serverPort, monitorPort, agentNumber, robotType, teamName) {
        runStatistics.open("run_statistics.csv");
        fallRate.open("fall_rate.txt");
        episodeNumber = 0;
}

State ForwardWalkNeuralWalkLearningAgent::newEpisode() {
    episodeNumber++;
    std::cout << "Episode: " << episodeNumber << std::endl;
    // starting position
    wiz.setGameTime(0);
    wiz.setPlayMode(representations::RawPlayMode::PLAY_ON);
    //setAgentRandomPosition();
    wiz.setAgentPositionAndDirection(
            agentNumber,
            representations::PlaySide::LEFT,
            Vector3<double>(INITIAL_AGENT_POS.x, INITIAL_AGENT_POS.y, 0.35),
            0.0);

    wiz.setBallPosition(Vector3<double>(100.0, 0.0, 0.35));
    lastTranslations = queue<Vector3<double> > ();
    for (int i = 0 ; i < N_STEPS_REWARD; i++) {
        lastTranslations.push(INITIAL_AGENT_POS);
    }

    maxVelocity = 0;
    nbSteps = 0;
    // xDistance = yDistance = maxHeight = 0.000;

    control.RestartController();

        

    decisionMaking.movementRequest = nullptr;
    desiredJoints = vector<double> ({0, 0, M_PI_2, 0, -M_PI_2, -M_PI_2, M_PI_2, 0, M_PI_2, M_PI_2, 
        0.0, 0.0, -M_PI/6.0, M_PI/3.0, -M_PI/6.0, 0.0, 0.0, 0.0, -M_PI/6.0 , M_PI/3.0, -M_PI/6.0, 0.0});

    for (int i = 0; i < 60; i++ ) {
        stepControl();
    }

    previousJoints = perception.getAgentPerception().getNaoJoints()
                    * NaoJoints::getNaoDirectionsFixing();
    previousOrientation = wiz.getAgentPose3D().rotation.getZAngle() - M_PI/2;
    previousHeight = wiz.getAgentTranslation().z;
    previousCoM = modeling.getAgentModel().getCenterOfMass();
    previousZMP = modeling.getAgentModel().getZeroMomentPoint();
    previousTorsoVel = modeling.getAgentModel().getTorsoAngularVelocity();
    previousTorsoAccel = modeling.getAgentModel().getTorsoAcceleration();
    previousLeftFootForceCoord = perception.getAgentPerception().getLeftFootForceResistanceData().getOriginCoordinates();
    previousLeftFootForce = perception.getAgentPerception().getLeftFootForceResistanceData().getForce();
    previousRightFootForceCoord = perception.getAgentPerception().getRightFootForceResistanceData().getOriginCoordinates();
    previousRightFootForce = perception.getAgentPerception().getRightFootForceResistanceData().getForce();
    rightFootCounter = 0;
    leftFootCounter = 0;
    rightFootWasTouchingGround = modeling.getAgentModel().rightFootTouchsGround();
    leftFootWasTouchingGround = modeling.getAgentModel().leftFootTouchsGround();
    return state();
}

SimulationResponse ForwardWalkNeuralWalkLearningAgent::runStep(Action action) {
    runStatistics << episodeNumber << "," << wiz.getAgentTranslation().x << "," << wiz.getAgentTranslation().y << std::endl;

    desiredJoints.clear();
    desiredJoints.push_back(0.0); //neckYaw
    desiredJoints.push_back(0.0); //neckPitch

    auto jointsMaxDeg = representations::TeamConstants::JOINTS_MAX_DEG.getAsVector();
    auto jointsMinDeg = representations::TeamConstants::JOINTS_MIN_DEG.getAsVector();
    for (int i = 0; i < NUMBER_OF_ACTION_DIM; i++) {
        double x = action.action(i);
        x = std::min(x, 1.0);
        x = std::max(x, -1.0);
        x = (x + 1.0) * ((jointsMaxDeg[i + 2] - jointsMinDeg[i + 2]) / 2.0) + jointsMinDeg[i + 2];
        x *= M_PI / 180.0;
        desiredJoints.push_back(x);
    }    


    ////////////////////////
    // SOCCER 3D SIMULATION STEP
    // Uses same input for some steps
    for (int i = 0; i < NUM_STEP_SAME_INPUT; i++) {
        stepControl();
    }

    nbSteps++;

    SimulationResponse stepUpdate;
    stepUpdate.mutable_state()->CopyFrom(state());
    stepUpdate.set_reward(reward());
    stepUpdate.set_done(episodeOver());

    return stepUpdate;
}

SetupEnvResponse ForwardWalkNeuralWalkLearningAgent::setup() {
    SetupEnvResponse initialInformation;
    initialInformation.set_num_state_dim(NUMBER_OF_STATE_DIM);
    initialInformation.set_num_action_dim(NUMBER_OF_ACTION_DIM);

    // Limits for actions
    // x speed
    for (int i = 0; i < NUMBER_OF_ACTION_DIM; i++)
        initialInformation.add_action_bound(M_PI);

    return initialInformation;
}

bool ForwardWalkNeuralWalkLearningAgent::episodeOver() {
    //episode achieved its length or agent has fallen
    if (wiz.getAgentTranslation().x > 14.0 || wiz.getAgentTranslation().z < 0.27) {
        fallRate << (wiz.getAgentTranslation().z < 0.33 ? 1 : 0) <<  std::endl;
        
        decisionMaking.movementRequest = nullptr;
        for (int i = 0; i < 80; i++ )
            step();
        
        std::cout << "Average Velocity: " << (wiz.getAgentTranslation().x  - INITIAL_AGENT_POS.x) / (nbSteps * STEP_TIME) << std::endl;
        std::cout << "Max Velocity: " << maxVelocity << std::endl;
        return true;
    }

    return false;
}

double ForwardWalkNeuralWalkLearningAgent::reward() {
    if (lastTranslations.size() != N_STEPS_REWARD) {
        lastTranslations.push(wiz.getAgentTranslation());
    } else {
        lastTranslations.pop();
        lastTranslations.push(wiz.getAgentTranslation());
    }

    double reward = 100.0*(lastTranslations.back().x - lastTranslations.front().x);

    double instVelocity = (lastTranslations.back().x - lastTranslations.front().x) / STEP_TIME;
    maxVelocity = std::max(maxVelocity, instVelocity);

    return reward;
}

State ForwardWalkNeuralWalkLearningAgent::state()  {
    State st;

    representations::NaoJoints currJoints = perception.getAgentPerception().getNaoJoints()
                    * NaoJoints::getNaoDirectionsFixing();

    //Counter
    st.add_observation(nbSteps);

    //Joints Information
    st.add_observation(currJoints.leftShoulderPitch);
    st.add_observation(currJoints.leftShoulderYaw);
    st.add_observation(currJoints.leftArmRoll);
    st.add_observation(currJoints.leftArmYaw);
    st.add_observation(currJoints.rightShoulderPitch);
    st.add_observation(currJoints.rightShoulderYaw);
    st.add_observation(currJoints.rightArmRoll);
    st.add_observation(currJoints.rightArmYaw);
    st.add_observation(currJoints.leftHipYawPitch);
    st.add_observation(currJoints.leftHipRoll);
    st.add_observation(currJoints.leftHipPitch);
    st.add_observation(currJoints.leftKneePitch);
    st.add_observation(currJoints.leftFootPitch);
    st.add_observation(currJoints.leftFootRoll);
    st.add_observation(currJoints.rightHipYawPitch);
    st.add_observation(currJoints.rightHipRoll);
    st.add_observation(currJoints.rightHipPitch);
    st.add_observation(currJoints.rightKneePitch);
    st.add_observation(currJoints.rightFootPitch);
    st.add_observation(currJoints.rightFootRoll);
    
    jointsDerivatives = currJoints - previousJoints;

    st.add_observation(jointsDerivatives.leftShoulderPitch);
    st.add_observation(jointsDerivatives.leftShoulderYaw);
    st.add_observation(jointsDerivatives.leftArmRoll);
    st.add_observation(jointsDerivatives.leftArmYaw);
    st.add_observation(jointsDerivatives.rightShoulderPitch);
    st.add_observation(jointsDerivatives.rightShoulderYaw);
    st.add_observation(jointsDerivatives.rightArmRoll);
    st.add_observation(jointsDerivatives.rightArmYaw);
    st.add_observation(jointsDerivatives.leftHipYawPitch);
    st.add_observation(jointsDerivatives.leftHipRoll);
    st.add_observation(jointsDerivatives.leftHipPitch);
    st.add_observation(jointsDerivatives.leftKneePitch);
    st.add_observation(jointsDerivatives.leftFootPitch);
    st.add_observation(jointsDerivatives.leftFootRoll);
    st.add_observation(jointsDerivatives.rightHipYawPitch);
    st.add_observation(jointsDerivatives.rightHipRoll);
    st.add_observation(jointsDerivatives.rightHipPitch);
    st.add_observation(jointsDerivatives.rightKneePitch);
    st.add_observation(jointsDerivatives.rightFootPitch);
    st.add_observation(jointsDerivatives.rightFootRoll);

    previousJoints = currJoints;

    st.add_observation(getAgentAngle());

    auto orientationDerivative = getAgentAngle() - previousOrientation;

    // st.add_observation(poseDerivative.translation.x);
    // st.add_observation(poseDerivative.translation.y);
    st.add_observation(orientationDerivative);

    previousOrientation = getAgentAngle();

    //Height Information
    st.add_observation(wiz.getAgentTranslation().z);

    heightDerivative = wiz.getAgentTranslation().z - previousHeight;

    st.add_observation(heightDerivative);

    previousHeight = wiz.getAgentTranslation().z;

    //Center of Mass Information

    st.add_observation(modeling.getAgentModel().getCenterOfMass().x);
    st.add_observation(modeling.getAgentModel().getCenterOfMass().y);
    st.add_observation(modeling.getAgentModel().getCenterOfMass().z);

    auto centerOfMassDerivative = modeling.getAgentModel().getCenterOfMass() - previousCoM;

    st.add_observation(centerOfMassDerivative.x);
    st.add_observation(centerOfMassDerivative.y);
    st.add_observation(centerOfMassDerivative.z);

    previousCoM = modeling.getAgentModel().getCenterOfMass();

    previousZMP = modeling.getAgentModel().getZeroMomentPoint();

    //Torso Angular Velocity Information

    st.add_observation(modeling.getAgentModel().getTorsoAngularVelocity().x);
    st.add_observation(modeling.getAgentModel().getTorsoAngularVelocity().y);
    st.add_observation(modeling.getAgentModel().getTorsoAngularVelocity().z);

    auto torsoVelocityDerivative = modeling.getAgentModel().getTorsoAngularVelocity() - previousTorsoVel;

    st.add_observation(torsoVelocityDerivative.x);
    st.add_observation(torsoVelocityDerivative.y);
    st.add_observation(torsoVelocityDerivative.z);

    previousTorsoVel = modeling.getAgentModel().getTorsoAngularVelocity();

    //Torso Acceleration Information

    st.add_observation(modeling.getAgentModel().getTorsoAcceleration().x);
    st.add_observation(modeling.getAgentModel().getTorsoAcceleration().y);
    st.add_observation(modeling.getAgentModel().getTorsoAcceleration().z);

    auto torsoAccelDerivative = modeling.getAgentModel().getTorsoAcceleration() - previousTorsoAccel;

    st.add_observation(torsoAccelDerivative.x);
    st.add_observation(torsoAccelDerivative.y);
    st.add_observation(torsoAccelDerivative.z);

    previousTorsoAccel = modeling.getAgentModel().getTorsoAcceleration();

    st.add_observation(perception.getAgentPerception().getLeftFootForceResistanceData().getOriginCoordinates().x);
    st.add_observation(perception.getAgentPerception().getLeftFootForceResistanceData().getOriginCoordinates().y);
    st.add_observation(perception.getAgentPerception().getLeftFootForceResistanceData().getOriginCoordinates().z);

    auto leftFootForceCoordDerivatives = perception.getAgentPerception().getLeftFootForceResistanceData().getOriginCoordinates() -
                previousLeftFootForceCoord;

    st.add_observation(leftFootForceCoordDerivatives.x);
    st.add_observation(leftFootForceCoordDerivatives.y);
    st.add_observation(leftFootForceCoordDerivatives.z);

    previousLeftFootForceCoord = perception.getAgentPerception().getLeftFootForceResistanceData().getOriginCoordinates();

    st.add_observation(perception.getAgentPerception().getLeftFootForceResistanceData().getForce().x);
    st.add_observation(perception.getAgentPerception().getLeftFootForceResistanceData().getForce().y);
    st.add_observation(perception.getAgentPerception().getLeftFootForceResistanceData().getForce().z);
    
    auto leftFootForceDerivatives = perception.getAgentPerception().getLeftFootForceResistanceData().getForce() -
                previousLeftFootForce;

    st.add_observation(leftFootForceDerivatives.x);
    st.add_observation(leftFootForceDerivatives.y);
    st.add_observation(leftFootForceDerivatives.z);

    previousLeftFootForce = perception.getAgentPerception().getLeftFootForceResistanceData().getForce();

    st.add_observation(perception.getAgentPerception().getRightFootForceResistanceData().getOriginCoordinates().x);
    st.add_observation(perception.getAgentPerception().getRightFootForceResistanceData().getOriginCoordinates().y);
    st.add_observation(perception.getAgentPerception().getRightFootForceResistanceData().getOriginCoordinates().z);

    auto rightFootForceCoordDerivatives = perception.getAgentPerception().getRightFootForceResistanceData().getOriginCoordinates() -
                previousRightFootForceCoord;

    st.add_observation(rightFootForceCoordDerivatives.x);
    st.add_observation(rightFootForceCoordDerivatives.y);
    st.add_observation(rightFootForceCoordDerivatives.z);

    previousRightFootForceCoord = perception.getAgentPerception().getRightFootForceResistanceData().getOriginCoordinates();

    st.add_observation(perception.getAgentPerception().getRightFootForceResistanceData().getForce().x);
    st.add_observation(perception.getAgentPerception().getRightFootForceResistanceData().getForce().y);
    st.add_observation(perception.getAgentPerception().getRightFootForceResistanceData().getForce().z);

    auto rightFootForceDerivatives = perception.getAgentPerception().getRightFootForceResistanceData().getForce() -
                previousRightFootForce;

    st.add_observation(rightFootForceDerivatives.x);
    st.add_observation(rightFootForceDerivatives.y);
    st.add_observation(rightFootForceDerivatives.z);

    previousRightFootForce = perception.getAgentPerception().getRightFootForceResistanceData().getForce();
    
    rightFootCounter++;
    leftFootCounter++;

    if (!leftFootWasTouchingGround && modeling.getAgentModel().leftFootTouchsGround()) {
        leftFootCounter = 0;
        leftFootWasTouchingGround = true;
    } else if (!modeling.getAgentModel().leftFootTouchsGround()) {
        leftFootWasTouchingGround = false;
    }

    if (!rightFootWasTouchingGround && modeling.getAgentModel().rightFootTouchsGround()) {
        rightFootCounter = 0;
        rightFootWasTouchingGround = true;
    } else if (!modeling.getAgentModel().rightFootTouchsGround()) {
        rightFootWasTouchingGround = false;
    }

    st.add_observation(rightFootCounter);
    st.add_observation(leftFootCounter);

    return st;
}

void ForwardWalkNeuralWalkLearningAgent::stepControl() {
    control.control(perception, desiredJoints);
    action.act(decisionMaking, control);
    communication.sendMessage(action.getServerMessage());
    communication.receiveMessage();
    perception.perceive(communication);
    modeling.model(perception, control);
}

void ForwardWalkNeuralWalkLearningAgent::step() {
    control.control(perception, modeling, decisionMaking);
    action.act(decisionMaking, control);
    communication.sendMessage(action.getServerMessage());
    communication.receiveMessage();
    perception.perceive(communication);
    modeling.model(perception, control);
}