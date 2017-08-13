#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <memory.h>
#include "timing.h"

#define LOCATIONNO 48
#define DATAFILE "locations.txt"

/*
	Xinyu Hong 05/09/2017
	Solving Travelling Salesman Problem by Genetic Algorithm
*/

typedef struct snp {
	float longitude;
	float latitude;
} location;
MPI_Datatype MPI_Loc;

typedef int path[LOCATIONNO];

location locs[LOCATIONNO];

void master();
void worker();
int rank, size;
double wc0, wc1, ct;

int main(int argc, char **argv)
{
	int i;
	int err;
	location aloc;
	int lenc[2];
	MPI_Aint locc[2];
	MPI_Datatype typc[2];
	MPI_Aint baseaddr;

	//Initialize MPI
	err = MPI_Init(&argc, &argv);
	err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	err = MPI_Comm_size(MPI_COMM_WORLD, &size);

	//Register MPI type
	MPI_Address(&aloc, &baseaddr);
	lenc[0] = 1;
	MPI_Address(&aloc.longitude, &locc[0]);
	locc[0] -= baseaddr;
	typc[0] = MPI_FLOAT;
	lenc[1] = 1;
	MPI_Address(&aloc.latitude, &locc[1]);
	locc[1] -= baseaddr;
	typc[1] = MPI_FLOAT;
	MPI_Type_struct(2, lenc, locc, typc, &MPI_Loc);
	MPI_Type_commit(&MPI_Loc);

	if (rank == 0) {
		master();
	} else {
		worker();
	}

	// Clean up
	err = MPI_Finalize();
	return 0;

}

void master(void)
{
	int i;
	FILE *flocations;
	if ((flocations = fopen(DATAFILE, "r")) == 0) {
		perror("locs file should be created");
		exit(1);
	}
	for (i = 0; i < LOCATIONNO; i++) {
		fscanf(flocations, "%f\t%f", &locs[i].longitude, &locs[i].latitude);
	}
	fclose(flocations);

	// start working as worker
	worker();
}

void random_shuffle(path p);
float fit(path p);
int comparepaths(const void *p1, const void *p2);
void crossover(path p1, path p2);
void printpath(path p);
void mutate(path p);
void select_crossover_choices(int *p, float *fits);
void random_crossover_choices(int *p, float *fits);

#define POPULATION_SIZE 2000
#define GENERATIONS 100
#define MUTATIONFACTOR 0.05
#define EXCHANGEPERIOD 10

void worker(void)
{
	timing(&wc0, &ct);
	int i, j;
	path *paths, *tmp;
	float *pathfits;
	struct {
		float value;
		int index;
	} in, out;
	struct timeval tv;
	int crossover_choices[2 * (POPULATION_SIZE - 2)];
	MPI_Status status;
	paths = (path *) malloc(POPULATION_SIZE * sizeof(path));
	pathfits = (float *) malloc(POPULATION_SIZE * sizeof(float));
	MPI_Bcast(locs, LOCATIONNO, MPI_Loc, 0, MPI_COMM_WORLD);
	// Initialize random number generator
	gettimeofday(&tv, NULL); //set up random seed based on time
	srandom(tv.tv_usec);
	
	// Initiate paths
	for (i = 0; i < LOCATIONNO; i++)
		paths[0][i] = i;
	for (i = 1; i < POPULATION_SIZE; i++) {
		memcpy(paths[i], paths[i - 1], sizeof(paths[i]));
		random_shuffle(paths[i]);
	}
	// Start evolution
	for (j = 0;; j++) {
		//sort paths
		qsort(*paths, POPULATION_SIZE, sizeof(paths[0]), comparepaths);
		if (j == GENERATIONS)
			break;
		//exchange
		if (j % EXCHANGEPERIOD == 0)
			MPI_Sendrecv(paths[POPULATION_SIZE - 1], LOCATIONNO, MPI_INT, (rank + 1) % size, 0, paths[0], LOCATIONNO, MPI_INT, (rank - 1) % size, 0, MPI_COMM_WORLD, &status);
		//evaluate fitness
		for (i = 0; i < POPULATION_SIZE; i++)
			pathfits[i] = fit(paths[i]);
		//create recombination order based on fitness score
		select_crossover_choices(crossover_choices, pathfits);
		//elitism
		for (i = 0; i < POPULATION_SIZE - 2; i++) {
			crossover(paths[crossover_choices[2 * i]], paths[crossover_choices[2 * i + 1]]);
		}
		//additional random recombination
		random_crossover_choices(crossover_choices, pathfits);
		for (i = 0; i < POPULATION_SIZE - 2; i += 2)
			crossover(paths[crossover_choices[i]], paths[crossover_choices[i + 1]]);
		for (i = 0; i < POPULATION_SIZE - 2; i ++)
			if (random() / RAND_MAX < MUTATIONFACTOR)
				mutate(paths[i]);
	}

	in.index = rank;
	in.value = fit(paths[POPULATION_SIZE - 1]);
	MPI_Allreduce(&in, &out, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
	//if the best path is in this worker
	timing(&wc1, &ct);
	if (rank == out.index) {
		printf("Best fit is:");
		printpath(paths[POPULATION_SIZE - 1]);
		printf("Elapsed time: %.2f\n", wc1-wc0);
	}

	free(pathfits);
	free(paths);

}

/* helper functions */
float sq(float x)
{
	return x * x;
}

float dist(location c1, location c2)
{
	return sqrt(sq(c1.longitude - c2.longitude) + sq(c1.latitude - c2.latitude));
}

/* crossover of two paths */
void crossover(path p1, path p2)
{
	int cp, i, j, k;
	path newp1, newp2;
	cp = random() % LOCATIONNO;
	for (i = 0; i < cp; i++)
		newp1[i] = p1[i];
	// from newp1[cp] on, add elements of p2 in p2 order
	j = 0;
	while (i < LOCATIONNO) {
		for (k = 0; k < i; k++)
			if (p2[j] == newp1[k])
				break;
		if (k == i)
			newp1[i++] = p2[j];
		j++ ;
	}
	for (i = 0; i < cp; i++)
		newp2[i] = p2[i];
	// from newp2[cp] on, add elements of p1 in p1 order
	j = 0;
	while (i < LOCATIONNO) {
		for (k = 0; k < i; k++)
			if (p1[j] == newp2[k])
				break;
		if (k == i)
			newp2[i++] = p1[j];
		j++;
	}
	memcpy(p1, newp1, sizeof(newp1));
	memcpy(p2, newp2, sizeof(newp2));
}

/* mutation of a path */
void mutate(path p)
{
	int c1, c2;
	int c;
	c1 = random() % LOCATIONNO;
	c2 = random() % LOCATIONNO;
	c = p[c1];
	p[c1] = p[c2];
	p[c2] = c;
}

/* shuffle, which is to mutate a path for random times */
void random_shuffle(path p)
{
	int i;
	for (i = 0; i < random() % LOCATIONNO; i++)
		mutate(p);
}

/* evaluate the fitness of the path*/
float fit(path p)
{
	int i;
	float sum = 0;
	for (i = 0; i < LOCATIONNO; i++)
		sum += dist(locs[p[i]], locs[p[(i + 1) % LOCATIONNO]]);
	return sum;
}

/*helper function for qsort: compare paths, the shorter the better*/
int comparepaths(const void *p1, const void *p2)
{
	float f1, f2;
	f1 = fit(*(path *) p1);
	f2 = fit(*(path *) p2);
	if (f1 < f2)
		return 1;
	else
		return (f1 == f2) ? 0 : -1;
}

/*print path*/
void printpath(path p)
{
	int i;
	for (i = 0; i < LOCATIONNO; i ++)
		printf("%d\t", p[i]);
	printf("fit=%f\n", fit(p));
}

void random_crossover_choices(int *p, float *fits)
{
	//randomly assign recombination orders
	int i;
	for (i = 0; i < POPULATION_SIZE - 2; i++)
		p[i] = i;
	for (i = 0; i < random() % (POPULATION_SIZE - 2); i++) {
		int c1, c2;
		int c;
		c1 = random() % (POPULATION_SIZE - 2);
		c2 = random() % (POPULATION_SIZE - 2);
		c = p[c1];
		p[c1] = p[c2];
		p[c2] = c;
	}
}

void select_crossover_choices(int *p, float *fits)
{
	int i, j;
	float tmp, sum = 0;
	//fits array is sorted largest first, and less is better
	tmp = fits[POPULATION_SIZE - 1];
	//implements the selection operation
	//calculate the probability by measuring how far each path is from the best path
	for (i = 0; i < POPULATION_SIZE; i++) {
		fits[i] = tmp - fits[i];
		sum += fits[i];
	}
	for (i = 0; i < POPULATION_SIZE; i++) {
		fits[i] /= sum;
	}
	for (i = POPULATION_SIZE - 2; i >= 0; i--) {
		fits[i] += fits[i + 1];
	}
	for (i = 0; i < 2 * (POPULATION_SIZE - 2); i++) {
		//get candidates for crossover
		float r = random() / RAND_MAX;
		for (j = 0; j < POPULATION_SIZE; j++)
			if (fits[j] < r)
				break;
		p[i] = j - 1;
	}
}
