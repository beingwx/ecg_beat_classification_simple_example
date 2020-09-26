#include <iostream>
#include <iomanip>
#include <vector>
#include <time.h>

#define CLOCKS_PER_SEC ((clock_t)1000)

#define COMPILER_MSVC

#define BATCH_SIZE 10
#define DATA_LEN 150
#define CLASS_NUM 2

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace std;

using namespace tensorflow;

void RunTensorflowSession();
bool LoadData(vector<vector<float> >& data);
bool NormalizeData(vector<float>& destDat, const short* srcDat, int len);
void ArgMax(vector<unsigned char>& classType, const tensorflow::TTypes<float, 1>::Tensor& prediction);

int main()
{
	clock_t start, finish;
	double duration;
	start = clock();

	RunTensorflowSession();

	finish = clock();
	cout << endl;
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "\n Total run time is:" << duration << " Seconds" << endl;

	return 0;
}

void RunTensorflowSession()
{
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		cout << status.ToString() << std::endl;
	}
	else {
		cout << "Session created successfully" << std::endl;
	}

	GraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), "ecg_model0.pb", &graph_def);
	if (!status.ok()) {
		cout << status.ToString() << std::endl;
	}
	else {
		cout << "Load graph protobuf successfully" << std::endl;
	}

	status = session->Create(graph_def);
	if (!status.ok()) {
		cout << status.ToString() << std::endl;
	}
	else {
		cout << "Add graph to session successfully" << std::endl;
	}

	vector<vector<float> > vDat;
	LoadData(vDat);


	Tensor input_tensor(DT_FLOAT, TensorShape({ BATCH_SIZE, DATA_LEN, 1 }));

	auto input_tensor_mapped = input_tensor.tensor<float, 3>();
	int i, j;
	for (i = 0; i < BATCH_SIZE; ++i)
	{
		for (j = 0; j < DATA_LEN; ++j)
		{
			input_tensor_mapped(i, j, 0) = vDat[i][j];
		}
	}


	vector<pair<string, Tensor>> inputs = {
		{ "Input:0", input_tensor },
	}; // input

	vector<Tensor> outputs; // output

	status = session->Run(inputs, { "Identity:0" }, {}, &outputs);
	if (!status.ok()) {
		cout << status.ToString() << std::endl;
	}
	else {
		cout << "\nRun session successfully\n" << std::endl;
	}

	cout << "output: \n";
	cout << outputs[0].DebugString() << std::endl;
	const tensorflow::TTypes<float, 1>::Tensor& prediction = outputs[0].flat_inner_dims<float, 1>();

	vector<unsigned char> classType;
	ArgMax(classType, prediction);


	session->Close();
}

bool LoadData(vector<vector<float>>& data)
{
	FILE* fp = fopen("data_test.dat", "rb");
	if (NULL == fp)
	{
		cout << "data load error!\n";
		return false;
	}

	data.clear();
	data.resize(BATCH_SIZE);

	short dat[DATA_LEN];
	for (int i = 0; i < BATCH_SIZE && !feof(fp); ++i)
	{
		data[i].resize(DATA_LEN, 0.0);

		fread(dat, sizeof(short), DATA_LEN, fp);

		NormalizeData(data[i], dat, DATA_LEN);
	}
	fclose(fp);
}

bool NormalizeData(vector<float>& destDat, const short* srcDat, int len)
{
	if ((NULL == srcDat) || (len != destDat.size()))
	{
		return false;
	}

	int i;
	short maxValue, minValue;
	maxValue = minValue = srcDat[0];
	for (i = 1; i < len; ++i)
	{
		destDat[i] = (float)srcDat[i];

		if (maxValue < srcDat[i])
		{
			maxValue = srcDat[i];
		}
		else if (minValue > srcDat[i])
		{
			minValue = srcDat[i];
		}
	}

	if (maxValue != minValue)
	{
		maxValue -= minValue;
		for (i = 0; i < len; ++i)
		{
			destDat[i] = (float)(srcDat[i] - minValue) / (float)maxValue;
		}
	}

	return true;
}

void ArgMax(vector<unsigned char>& classType, const tensorflow::TTypes<float, 1>::Tensor& prediction)
{
	classType.clear();

	int i, j;
	float maxValue = -1.0;
	int maxIndex = -1;
	const long count = prediction.size();

	cout << "\nThe prediction of test data: \n";

	for (i = 0; i < BATCH_SIZE; ++i)
	{
		maxValue = -1.0;
		maxIndex = -1;

		printf("\nbatch[%d]: ", i);

		for (j = 0; j < CLASS_NUM; ++j)
		{
			const float value = prediction(i * CLASS_NUM + j);
			
			cout << left << setw(12) << value;
			cout << (j < CLASS_NUM - 1 ? ", " : "\n");

			if (value > maxValue) {
				maxIndex = j;
				maxValue = value;
			}
		}

		classType.push_back(maxIndex);
	}
}