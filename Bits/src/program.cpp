#include <iostream>
#include <math.h>
#include <algorithm>

#define INPUT_LENGTH 5
#define NB_HIDDEN_NEURONS 3
#define TRAIN_SET_SIZE 6
#define NB_ITERATIONS 10

float _trainSet[TRAIN_SET_SIZE][INPUT_LENGTH] =
    {
        {0, 0, 1, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0},
        {1, 1, 0, 1, 1},
        {0, 1, 0, 1, 0},
        {0, 1, 1, 1, 0}};

float _trainExpected[TRAIN_SET_SIZE] = {1, 1, 0, 0, 0, 1};


float _tested[][INPUT_LENGTH] = {
    {0, 1, 0, 1, 0},
    {1, 0, 1, 0, 1},
    {1, 1, 1, 0, 0},
};

float _leftWeights[NB_HIDDEN_NEURONS][INPUT_LENGTH] = {};
float _rightWeights[NB_HIDDEN_NEURONS] = {};
float _hiddenNeurons[NB_HIDDEN_NEURONS] = {};
float _output;

float _hiddenCosts[NB_HIDDEN_NEURONS] = {};
float _outputCost;

float alpha = 0.1;

static void ForthPropagation(int entry);
static void BackPropagation(int entry);
static void InitializeWeights(void);
static void FillHiddenNeurons(int entry);
static void ComputeOutput(void);

static void ComputeHiddenNeurons(float inputs[]);

static float Sigmoid(float input);
static float SigmoidPrime(float input);

// Detect if middle bit is 1
int main(int argc, char *argv[])
{
    InitializeWeights();
    for (int it = 0; it < NB_ITERATIONS; it++)
    {
        for (int n = 0; n < TRAIN_SET_SIZE; n++)
        {
            ForthPropagation(n);
            BackPropagation(n);
        }
    }

    for (int i=0; i < 3; i++)
    {
        ComputeHiddenNeurons(_tested[i]);
        ComputeOutput();
        std::cout << "Wanted : " << _tested[i][3] << ", prediction : " << _output << std::endl;
    }
    return 0;
}

static void ForthPropagation(int n)
{
    FillHiddenNeurons(n);
    ComputeOutput();
}

static void BackPropagation(int n)
{
    _outputCost = pow(_trainExpected[n] - _output, 2);
    for (int h = 0; h < NB_HIDDEN_NEURONS; h++)
        _hiddenCosts[h] = _rightWeights[h] * _outputCost * SigmoidPrime(_hiddenNeurons[h]);

    // Reset left weights
    for (int i = 0; i < NB_HIDDEN_NEURONS; i++)
    {
        for (int j = 0; j < INPUT_LENGTH; j++)
        {
            _leftWeights[i][j] += alpha * _trainSet[n][j] * _hiddenCosts[i];
        }
    }

    // Reset right weights
    for (int h = 0; h < NB_HIDDEN_NEURONS; h++)
    {
        _rightWeights[h] += alpha * _hiddenNeurons[h] * _outputCost;
    }
}

static void InitializeWeights()
{
    srand(time(NULL));
    for (int i = 0; i < NB_HIDDEN_NEURONS; i++)
    {
        for (int j = 0; j < INPUT_LENGTH; j++)
            _leftWeights[i][j] = (float)rand() / RAND_MAX;
        _rightWeights[i] = (float)rand() / RAND_MAX;
    }
}

static void ComputeHiddenNeurons(float inputs[])
{
    for (int j = 0; j < NB_HIDDEN_NEURONS; j++)
    {
        float neuronValue = 0;
        for (int i = 0; i < INPUT_LENGTH; i++)
        {
            neuronValue += _leftWeights[j][i] * inputs[i];
        }
        _hiddenNeurons[j] = Sigmoid(neuronValue);
    }
}

static void FillHiddenNeurons(int n)
{
    for (int j = 0; j < NB_HIDDEN_NEURONS; j++)
    {
        float neuronValue = 0;
        for (int i = 0; i < INPUT_LENGTH; i++)
        {
            neuronValue += _leftWeights[j][i] * _trainSet[n][i];
        }
        _hiddenNeurons[j] = Sigmoid(neuronValue);
    }
}

static void ComputeOutput()
{
    float val = 0;
    for (int h = 0; h < NB_HIDDEN_NEURONS; h++)
        val += _rightWeights[h] * _hiddenNeurons[h];
    _output = Sigmoid(val);
}

static float Sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

static float SigmoidPrime(float x)
{
    return Sigmoid(x) * (1 - Sigmoid(x));
}