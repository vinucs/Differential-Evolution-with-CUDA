#include <iostream>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cuda.h>
#include <curand.h> 

#define M_PI 3.14159265358979323846
#define N_RAND_IND 3
#define MAX_FITNESS 4.25389
#define MUT_RAND_N 3
#define MIN_VALUE -1.0
#define MAX_VALUE 2.0


/* Maximizar a funcao: f(x,y) = x*sin(4*pi*x) - y*sin(4*pi*y + pi) + 1,
 e que o maximo global de 4.25389 situa-se no ponto (1.62888, 1.62888) */

__global__ void initializePopulation(double *pop, size_t pitch, int num_pop, int num_genes, double *randpop){
    int index_gen = blockIdx.x;
    int index_pop = (blockIdx.x*blockDim.x + threadIdx.x);
    if(index_pop >= 100)
        index_pop -= (100*index_gen);
    if(index_pop < num_pop && index_gen < num_genes){
        double *indv = (double *)((char*)pop + index_pop*pitch);
        double *rand_ind = (double *)((char*)randpop + index_pop*pitch);
        indv[index_gen] = MIN_VALUE + rand_ind[index_gen]*(MAX_VALUE - MIN_VALUE);
    }
}

__global__ void fitness(double *pop, size_t pitch, const int num_genes, double *result){
    int index_pop = (blockIdx.x*blockDim.x + threadIdx.x);
    double *indv = (double *)((char*)pop + index_pop*pitch);
    result[index_pop] = indv[0]*sin(4*M_PI*indv[0]) - indv[1]*sin(4*M_PI*indv[1] + M_PI) + 1;
    __syncthreads();
}

__host__ void getBestIndividual(double *pop, size_t pitch, const int num_pop, const int num_genes, int *best_index, double *best_score){
    double *d_scores;
    cudaMalloc((void**)&d_scores, num_pop*sizeof(double));
    fitness<<<1, num_pop>>>(pop, pitch, num_genes, d_scores);
    double h_scores[100];
    cudaMemcpy(h_scores, d_scores, num_pop*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_scores);
    *best_index = 0;
    *best_score = h_scores[*best_index];
    for(int i = 1; i < num_pop; i++){
        if(h_scores[i] > *best_score){
            *best_index = i;
            *best_score = h_scores[i];
        }
    }
}

__global__ void mutationRand(double *pop, double *new_pop, size_t pitch_pop, const int num_pop, const int num_genes, double mutation_factor, int *randmut, size_t pitch_mut){
    int index_gen = blockIdx.x;
    int index_pop = (blockIdx.x*blockDim.x + threadIdx.x);
    if(index_pop >= 100)
        index_pop -= (100*index_gen);
    if(index_pop < num_pop && index_gen < num_genes){
        int *rand_indvs = (int *)((char*)randmut + index_pop*pitch_mut);
        double *rand1 = (double *)((char*)pop + rand_indvs[0]*pitch_pop);
        double *rand2 = (double *)((char*)pop + rand_indvs[1]*pitch_pop);
        double *rand3 = (double *)((char*)pop + rand_indvs[2]*pitch_pop);
        double *new_indv = (double *)((char*)new_pop + index_pop*pitch_pop);
        double value = rand1[index_gen] + (mutation_factor *(rand2[index_gen] - rand3[index_gen]));
        if(value < MIN_VALUE)
    	    value = MIN_VALUE;
	    else if(value > MAX_VALUE)
		    value = MAX_VALUE;
        new_indv[index_gen] = value;
    }
    __syncthreads();
}

__host__ void startMutationRand(double *pop, double* new_pop, size_t pitch_pop, const int num_pop, const int num_genes, double mutation_factor, dim3 blkgenes, dim3 thrdpop){
    int h_randmut[100][MUT_RAND_N];
    for(int i = 0; i < num_pop; i++){
        do h_randmut[i][0] = num_pop*((double) rand()/ ((double)RAND_MAX + 1));while(h_randmut[i][0] == i);
        do h_randmut[i][1] = num_pop*((double) rand()/ ((double)RAND_MAX + 1));while(h_randmut[i][1] == i || h_randmut[i][1] == h_randmut[i][0]);
        do h_randmut[i][2] = num_pop*((double) rand()/ ((double)RAND_MAX + 1));while(h_randmut[i][2] == i || h_randmut[i][2] == h_randmut[i][0] || h_randmut[i][2] == h_randmut[i][1]);
    }
    int *d_randmut;
    size_t pitch_mut;
    cudaMallocPitch((void**)&d_randmut, &pitch_mut, MUT_RAND_N, num_pop);
    cudaMemcpy2D(d_randmut, pitch_mut, h_randmut, MUT_RAND_N*sizeof(int), MUT_RAND_N*sizeof(int), num_pop, cudaMemcpyHostToDevice);
    mutationRand<<<blkgenes, thrdpop>>>(pop, new_pop, pitch_pop, num_pop, num_genes, mutation_factor, d_randmut, pitch_mut);
    cudaFree(d_randmut);
}

// __device__ void f1(double *x,int D,double *out){
//     int i;
//     __shared__ double result[NUMTHREAD];
//     result[threadIdx.x]=0;
//     for (i=threadIdx.x;i<dim;i+=blockDim.x)
//     result[threadIdx.x]+= x[i]*__sinf(sqrtf(fabsf(x[i])));
//     sum(result,NUMTHREAD);
//     *out = result[0];
// }

// __global__ void generation_new_population (double *pop,int NP, int D, double *npop, double F, double CR,double *rand, int *mutation,double min,double max){
//     int first,last,a,b,c,k,i,j;
//     for(i=blockIdx.x;i<NP;i+=gridDim.x){
//         first=i*D;
//         last=first+D;
//         k=threadIdx.x;
//         a=mutation[i*3];
//         b=mutation[i*3+1];
//         c=mutation[i*3+2];
//         j= threadIdx.x;
//         for(j+=first;j<last;j+=blockDim.x,){
//             if(rand[j]<CR)
//                 npop[j]= pop[c+k]+F*(pop[a+k]-pop[b+k]);
//             else npop[j]=pop[j];
//             if(npop[j]>max) npop[j]=max;
//             else 
//                 if(npop[j]<min) npop[j]=min;
//             k+=blockDim.x;
//         }
//         __syncthreads();
//     }
// }

// __global__ void selection(int function ,int NP, int D, double *pop, double *npop, double *fobj){
//     int i,first,last,j;
//     double r;
//     for(i=blockIdx.x;i<NP;i+=gridDim.x){
//         first=i*D;
//         r=func(function, &npop[first], D);
//         __syncthreads();
//         if(r<fobj[i]){
//             first=i*D;
//             last=first+D;
//             first+=threadIdx.x;
//             for(int j=first;j<last;j+=blockDim.x)
//                 pop[j]=npop[j];
//             fobj[i]=r;
//         }
//         __syncthreads();
//     }
// }

// double **drawPopulationInit(double **d_ran){
//     double **randpop = new double*[num_population];
//     for (int i = 0; i < num_genes; i++){
//         population[i] = new double[num_genes];
//         new_population[i] = new double[num_genes];
//     }
// }

// void drawIndividuals(int **randpop, int num_pop){
//     for(int i = 0; i < num_pop; i++){
//         do{
//             rand_a = max_generation*((double)rand()/(double) RAND_MAX);
//         }while(rand_a == i);
//         do{
//             rand_b = max_generation*((double)rand()/(double) RAND_MAX);
//         }while(rand_b == i || rand_b == rand_a);
//         do{
//             rand_c = max_generation*((double)rand()/(double) RAND_MAX);
//         }while(rand_c == i || rand_c == rand_a || rand_b == rand_a);
//         randpop[i][0]=rand_a;
//         randpop[i][1]=rand_b;
//         randpop[i][2]=rand_c;
//     }
// }

__host__ void printPopulation(double **pop, int num_pop, int num_genes){
    for(int i = 0; i < num_pop; i++){
        for(int j = 0; j < num_genes; j++)
            std::cout << pop[i][j] << " ";
        std::cout << std::endl;
    }
}

int main(int argc, char **argv){
    //Differential Evolution parameters
    const int max_generation = 1;
    const int num_population = 100;
    const int num_genes = 2;
    const double mutation_factor = 0.2;
    const double crossover_rate = 0.5;

    //Setting block and grid lenght
    dim3 blocksforgenes(num_genes, 1, 1);
    dim3 threadsforpop(num_population, 1, 1);
    

    //Creating random seed generator by device
    // curandGenerator_t gen_rand;
    // curandCreateGenerator(&gen_rand, CURAND_RNG_PSEUDO_DEFAULT);
    // curandSetPseudoRandomGeneratorSeed(gen_rand, 1234ULL);

    //Creating random seed by host
    srand(1);

    //Setting population vector to host and device
    size_t pitch_pop;
    double h_population[num_population][num_genes];
    double h_newpopulation[num_population][num_genes];
    double *d_population, *d_newpopulation;
    cudaMallocPitch((void**)&d_population, &pitch_pop, num_genes, num_population);
    cudaMallocPitch((void**)&d_newpopulation, &pitch_pop, num_genes, num_population);

    //Draw rand(0,1) to initialize population
    double h_randpop[num_population][num_genes];
    for(int i = 0; i < num_population; i++){
        for(int j = 0; j < num_genes; j++)
            h_randpop[i][j] = ((double) rand()/ ((double)RAND_MAX + 1));
    }
    
    //Initialize population
    double *d_randpop;
    cudaMallocPitch((void**)&d_randpop, &pitch_pop, num_genes, num_population);
    cudaMemcpy2D(d_randpop, pitch_pop, h_randpop, num_genes*sizeof(double), num_genes*sizeof(double), num_population, cudaMemcpyHostToDevice);
    // for(int i = 0; i < num_population; i++)
    //     curandGenerateUniform(gen_rand, randpop[i], num_genes);
    initializePopulation<<<blocksforgenes,threadsforpop>>>(d_population, pitch_pop, num_population, num_genes, d_randpop);
    cudaFree(d_randpop);

    int generation, best_individual_index;
    double best_individual_score,  previous_best_score = DBL_MIN;
    for(generation = 1; generation <= max_generation; generation++){

        //Best individual
        getBestIndividual(d_population, pitch_pop, num_population, num_genes, &best_individual_index, &best_individual_score);
        if(previous_best_score < best_individual_score)
            printf("best individual index: %d - fitness: %f", best_individual_index, best_individual_score);

        //Stop condition
        if((MAX_FITNESS - best_individual_score) <= 0.00001){
            break;
        }

        //Mutation phase
        startMutationRand(d_population, d_newpopulation, pitch_pop, num_population, num_genes, mutation_factor, blocksforgenes, threadsforpop);

        //Crossover phase

        //Selection Phase

    }

    // for(int i = 0; i < max_generation; i++){
    //     int rand_a, rand_b, rand_c;
    //     cudaMemcpy(d_mutation, h_mutation, 3*NP*sizeof(int), cudaMemcpyHostToDevice);
    //     generation_new_population<<<32,64>>>(d_pop,NP,D,d_npop,F,CR,d_Rand,d_mutation,s_min,s_max);
    //     evaluate_new_population<<<32,64>>>(function,NP,D,d_pop,d_npop,d_fobj);
    // }
    // min_value_index<<<1,1>>>(0,NP,D,d_fobj,d_bestNP,d_pop,bestVal);
    // cudaMemcpy(h_best, d_best, D*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&h_bestVal, bestVal, sizeof(double), cudaMemcpyDeviceToHost);


    // cudaMemcpy2D(h_population, num_genes*sizeof(double), d_population, pitch_pop, num_genes*sizeof(double), num_population, cudaMemcpyDeviceToHost);
    // std::cout << "POPULATION\n";
    // for(int i = 0; i < num_population; i++){
    //     for(int j = 0; j < num_genes; j++)
    //         std::cout << h_population[i][j] << " ";
    //     std::cout << std::endl;
    // }
    // cudaMemcpy2D(h_newpopulation, num_genes*sizeof(double), d_newpopulation, pitch_pop, num_genes*sizeof(double), num_population, cudaMemcpyDeviceToHost);
    // std::cout << "MUTANTS:\n";
    // for(int i = 0; i < num_population; i++){
    //     for(int j = 0; j < num_genes; j++)
    //         std::cout << h_newpopulation[i][j] << " ";
    //     std::cout << std::endl;
    // }

    //Free memory from host and device
    cudaFree(d_population);
    cudaFree(d_newpopulation);
    // curandDestroyGenerator(gen_rand);
}