#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>

#include "particle_filter.h"

using namespace std;
#define _USE_MATH_DEFINES

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	num_particles = 50;
	
    particles.resize(num_particles);
    weights.resize(num_particles);

	double initial_weight = 1.0 / static_cast<double>(num_particles);

	random_device rd;
    default_random_engine gen(rd());

	normal_distribution<double> dist_x(x, std[0]);	
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	for(int i = 0; i<num_particles;i++){
		Particle &p = particles[i];
		double &w = weights[i];
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = initial_weight;
		w = initial_weight;
	}
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	random_device rd;
	default_random_engine gen(rd());
	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);

	double tmp_x;
	double tmp_y;
	double tmp_theta;
	for(int p_index = 0; p_index<num_particles; p_index++){

		Particle &p = particles[p_index];
	
		// Could be part of a separate function
		if(fabs(yaw_rate) < 0.0001){
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		}
		else{
			double speed_yawrate_ratio = velocity/yaw_rate;
			double yaw_rate_influence = yaw_rate*delta_t;
			
            const double theta_n = p.theta + yaw_rate * delta_t;
			p.x += (velocity/yaw_rate) * ( sin(theta_n) - sin(p.theta));
            p.y += (velocity/yaw_rate) * (-cos(theta_n) + cos(p.theta));
            p.theta = p.theta + yaw_rate * delta_t;
		}

        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
 	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs> &observations) {
	
	vector<LandmarkObs> closest_landmarks(observations.size());
	
	default_random_engine generator;
	bernoulli_distribution distribution(0.5);
	
	for(int i = 0; i < observations.size();i++){
		LandmarkObs closest_landmark;
		double min_distance = 0.0;
		for(int j = 0;j<predicted.size();j++){
			double tmp_dist = dist(observations[i].x,observations[i].y, predicted[j].x,predicted[j].y);
			if(tmp_dist<min_distance){
				min_distance = tmp_dist;
				closest_landmark = predicted[j];
			}
			if(tmp_dist==min_distance && distribution(generator))
				closest_landmark = predicted[j];
		}
		closest_landmarks[i] = closest_landmark;
	}	
	observations = closest_landmarks;
}

/**
 * This function compute the NN map landmark respect to the observed landmarks
 * @params {Map} map the map object with all the map landmarks
 * {LandmarkObs} observation the observed landmarks in map coordinates from which find the closest map landmark
 * @return {Map::single_landmark_s} the closest map landmark in the map single_landmark_s form
 */
inline Map::single_landmark_s findNNLandmark(Map map,LandmarkObs observation) {
	
	default_random_engine generator;
	bernoulli_distribution distribution(0.5);
	
	Map::single_landmark_s closest_landmark = map.landmark_list[0];
	
	double min_dist = dist(observation.x,observation.y, closest_landmark.x_f,closest_landmark.y_f);
	for(size_t j = 1;j<map.landmark_list.size();j++){
		
		Map::single_landmark_s current_landmark = map.landmark_list[j];
		double cur_dist = dist(observation.x,observation.y, current_landmark.x_f,current_landmark.y_f);
		if(cur_dist<min_dist || (cur_dist==min_dist && distribution(generator))){
			closest_landmark = map.landmark_list[j];	
			min_dist = cur_dist;
		}
	}
	return closest_landmark;
}

/**
 * This will transform an observed landmark from the particle coordinate system, to the map reference system
 * @params {Particle} p the particle from which compute the transformation
 * {LandmarkObs} ob the observed landmark in particle reference system
 * @return {LandmarkObs} the observed landmark in the map reference system
 */
inline LandmarkObs fromVehicle2MapCoordinates(Particle p, LandmarkObs ob){
	
	LandmarkObs NN_landmark;
	
	NN_landmark.id = ob.id;
	NN_landmark.x = p.x + ob.x*cos(p.theta) - ob.y*sin(p.theta);
	NN_landmark.y = p.y + ob.x*sin(p.theta) + ob.y*cos(p.theta);

	return NN_landmark;	
}

/**
 * This function compute the influence of a single observed landmark with respect to the NN map landmark, to a specific particle weight
 * @params 
 * {double} sigma_x is the sigma_x variance along the x measurement axis
 * {double} sigma_y is the sigma_y variance along the y measurement axis
 * {Map::single_landmark_s} closest_landmark the NN landmark to the observation currently under verification
 * {LandmarkObs} obs_map_coords the observed landmark currently under verification
 * @return 
 * {double} the closest map landmark in the map single_landmark_s form
 */
inline double computeObsWeightInfluence(double sigma_x, double sigma_y,Map::single_landmark_s NN_landmark, LandmarkObs obs_landmark){

	double den = 1/(2.0*M_PI*sigma_x*sigma_y);
	double exp_arg = pow((NN_landmark.x_f - obs_landmark.x),2) / (pow(sigma_x,2))+
		pow((NN_landmark.y_f - obs_landmark.y),2) / (pow(sigma_y,2));

	return den*exp(-0.5*exp_arg);

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], vector<LandmarkObs> observations, Map map_landmarks) {

	double sum_weights = 0.0;
	
	for(int i = 0;i<num_particles;i++){

		vector<LandmarkObs> observations_in_map_coordinates(observations.size());
		vector<Map::single_landmark_s> closest_map_landmarks(map_landmarks.landmark_list.size());
		double weight = 1.0;
		for (int j = 0; j <  observations.size();j++){
			
			LandmarkObs obs_landmark = fromVehicle2MapCoordinates(particles[i],observations[j]);
			Map::single_landmark_s NN_landmark = findNNLandmark(map_landmarks, obs_landmark);

			weight *= computeObsWeightInfluence(std_landmark[0],std_landmark[1],NN_landmark, obs_landmark);
		}
		
		sum_weights += weight;
    	particles[i].weight = weight;
	}
	for(int i = 0; i<num_particles;i++){
		particles[i].weight /= sum_weights;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample()
{

    // Resample according to weights
    std::vector<Particle> particles_original(particles);
    std::default_random_engine gen;
    std::discrete_distribution<std::size_t> d(weights.begin(), weights.end());

    for (Particle& p : particles)
    {
        p = particles_original[d(gen)];
    }
}

void ParticleFilter::write(string filename) {
	ofstream dataFile;
	dataFile.open(filename, ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
