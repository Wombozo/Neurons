#include <iostream>
#include <math.h>
#include <algorithm>

#define INPUT_LENGTH 5
#define NB_HIDDEN_NEURONS 3

float _trainSet[6][INPUT_LENGTH] =
    {
        {0, 0, 1, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0},
        {1, 1, 0, 1, 1},
        {0, 1, 0, 1, 0},
        {0, 1, 1, 1, 0}};

float _trainExpected[6] = {1, 1, 0, 0, 0, 1};

float _leftWeights[NB_HIDDEN_NEURONS][INPUT_LENGTH] = {};
float _rightWeights[NB_HIDDEN_NEURONS] = {};
float _hiddenNeurons[NB_HIDDEN_NEURONS] = {};
float _output;

float _hiddenCosts[NB_HIDDEN_NEURONS] = {};
float _outputCost;

float alpha = 0.1;

static void ForthPropagation(void);
static void BackPropagation(void);
static void InitializeWeights(void);
static void FillHiddenNeurons(void);
static void ComputeOutput(void);

static float Sigmoid(float input);

// Detect if middle bit is 1
int main(int argc, char *argv[])
{
    InitializeWeights();
    ForthPropagation();
    BackPropagation();
    return 0;
}

static void ForthPropagation()
{
    FillHiddenNeurons();
    ComputeOutput();
    std::cout << "  Output = " << _output << std::endl;
}

static void BackPropagation()
{
    _outputCost = pow(_trainExpected[0] - _output,2);
    for (int h = 0; h < NB_HIDDEN_NEURONS; h++)
        _hiddenCosts[h] = _rightWeights[h] * _outputCost * SigmoidPrime(_hiddenNeurons[h]);

    // Reset left weights
    for (int i = 0; i < NB_HIDDEN_NEURONS; i++)
    {
        for (int j=0; j < INPUT_LENGTH; j++)
        {
            _leftWeights[i][j] += alpha * _trainSet[0][j] * _hiddenCosts[i];
        }
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

static void FillHiddenNeurons()
{
    for (int j = 0; j < NB_HIDDEN_NEURONS; j++)
    {
        float neuronValue = 0;
        for (int i = 0 ; i < INPUT_LENGTH; i++)
        {
            neuronValue += _leftWeights[j][i] * _trainSet[0][i];
        }
        _hiddenNeurons[j] = Sigmoid(neuronValue);
        std::cout << "NEURON " << j << ": " << _hiddenNeurons[j] << std::endl;
    }
}

static void ComputeOutput()
{
    float val=0;
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
    return Sigmoid(x)*(1-Sigmoid(x));
}