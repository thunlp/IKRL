
#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include<cstdlib>
#include<sstream>
#include<pthread.h>
#include<ctime>
#include<sys/time.h>
#include<assert.h>
using namespace std;

//Ruobing Xie
//Image-embodied Knowledge Representation Learning
//IKRL (ATT)

#define pi 3.1415926535897932384626433832795
#define THREADS_NUM 8

bool L1_flag=1;

int nepoch = 15000;		//iteration times
string init_version = "unif";		//unif/bern

//normal distribution
double rand(double min, double max)
{
    return min+(max-min)*rand()/(RAND_MAX+1.0);
}

double normal(double x, double miu,double sigma)
{
    return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

double randn(double miu,double sigma, double min ,double max)
{
    double x,y,dScope;
    do{
        x=rand(min,max);
        y=normal(x,miu,sigma);
        dScope=rand(0.0,normal(miu,miu,sigma));
    }while(dScope>y);
    return x;
}

double sqr(double x)
{
    return x*x;
}

double vec_len(vector<double> &a)
{
	double res=0;
    for (int i=0; i<a.size(); i++)
		res+=a[i]*a[i];
	res = sqrt(res);
	return res;
}

double norm(vector<double> &a)
{
	double x = vec_len(a);
	if (x>1)
		for (int ii=0; ii<a.size(); ii++)
			a[ii]/=x;
	return 0;
}

int rand_max(int x)
{
	int res = (rand()*rand())%x;
	while (res<0)
		res+=x;
	return res;
}

//parameters
string version;
char buf[100000],buf1[100000];
int relation_num,entity_num;		//number
map<string,int> relation2id,entity2id;
map<int,string> id2entity,id2relation;
int nbatches, batchsize;

map<int,map<int,int> > left_entity,right_entity;
map<int,double> left_num,right_num;

int n,method;		//n: dimension of entity/relation
double res_triple, res_ii, res_ie, res_ei, res_normal;		//loss function value
double res_thread_triple[THREADS_NUM], res_thread_ii[THREADS_NUM], res_thread_ie[THREADS_NUM], res_thread_ei[THREADS_NUM], res_thread_normal[THREADS_NUM];		//loss for each thread
double count,count1;
double rate,rate_end,rate_begin;		//learning rate
double rate_n,rate_m,rate_i;
double margin;		//margin
double belta;
vector<int> fb_h,fb_l,fb_r;
vector<vector<double> > relation_vec,entity_vec;		//embeddings
vector<vector<double> > relation_tmp,entity_tmp;

vector<int> image_count_vec;		//image_num for each entity
vector<vector<vector<double> > > image_vec;		//ent_num*img_num*img_dim, image feature vector
vector<vector<double> > image_att_vec;		//ent_num*img_dim, aggregated image-based representation for each entity via mean/max/attention
vector<vector<double> > image_proj_vec;		//ent_num*ent_dim, projection of image_vec in entity space
vector<vector<double> > image_proj_grad;		//ent_num*ent_dim, gradient
vector<int> image_proj_state;		//ent_num, update mark

vector<vector<double> > image_mat;		//ent_dim*img_dim, projection matrix, image(4096) -> entity(n)
vector<vector<vector<double> > > image_mat_tmp;

vector<double> posErrorVec[THREADS_NUM];
vector<double> negErrorVec[THREADS_NUM];

map<pair<int,int>, map<int,int> > ok;		//positive mark

pthread_mutex_t mut_mutex;

void sgd();
void train_ee(int, int, int, int, int, int, int);
void train_ii(int, int, int, int, int, int, int);
void train_ei(int, int, int, int, int, int, int);
void train_ie(int, int, int, int, int, int, int);
void calc_image_proj_vec(int);
void *rand_sel(void *);
void update_multithread();

//train
void run(int n_in,double rate_in,double margin_in,int method_in)
{
	n = n_in;
	rate = rate_in;
	rate_i = 1.0*rate_in;		//different learning rate for different entity representation type combination
	rate_m = 0.5*rate_in;
	rate_n = 0.2*rate_in;
	margin = margin_in;
	method = method_in;
	
	//relation & entity initialization
	relation_vec.resize(relation_num);
	for (int i=0; i<relation_vec.size(); i++)
		relation_vec[i].resize(n);
	entity_vec.resize(entity_num);
	for (int i=0; i<entity_vec.size(); i++)
		entity_vec[i].resize(n);
	relation_tmp.resize(relation_num);
	for (int i=0; i<relation_tmp.size(); i++)
		relation_tmp[i].resize(n);
	entity_tmp.resize(entity_num);
	for (int i=0; i<entity_tmp.size(); i++)
		entity_tmp[i].resize(n);
	
	/*
	//init randomly
	for (int i=0; i<relation_num; i++)
	{
		for (int ii=0; ii<n; ii++)
			relation_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
		norm(relation_vec[i]);
	}
	for (int i=0; i<entity_num; i++)
	{
		for (int ii=0; ii<n; ii++)
			entity_vec[i][ii] = randn(0,1.0/n,-6/sqrt(n),6/sqrt(n));
		norm(entity_vec[i]);
	}
	*/
	
	//or init by pre-trained entity/relation embeddings
	FILE* f1 = fopen(("../init_res/entity2vec."+init_version).c_str(),"r");
	for (int i=0; i<entity_num; i++)
	{
		for (int ii=0; ii<n; ii++)
			fscanf(f1,"%lf",&entity_vec[i][ii]);
		norm(entity_vec[i]);
	}
	fclose(f1);

	FILE* f2 = fopen(("../init_res/relation2vec."+init_version).c_str(),"r");
	for (int i=0; i<relation_num; i++)
	{
		for (int ii=0; ii<n; ii++)
			fscanf(f2,"%lf",&relation_vec[i][ii]);
	}
	fclose(f2);
	
	
	//image projection matrix initialization
	
	//init randomly
	image_mat.resize(n);
	for(int i=0; i<n; i++)
	{
		image_mat[i].resize(4096);
		for(int ii=0; ii<4096; ii++)
		{
			image_mat[i][ii] = randn(0,1.0/4096,-6/sqrt(4096),6/sqrt(4096));
		}
	}
	
	/*
	//or init by pre-trained results
	FILE* f3 = fopen(("../init_res/image_mat."+init_version).c_str(),"r");
	image_mat.resize(n);
	for(int i=0; i<n; i++)
	{
		image_mat[i].resize(4096);
		for(int ii=0; ii<4096; ii++)
		{
			fscanf(f3,"%lf",&image_mat[i][ii]);
		}
	}
	fclose(f3);
	*/
	
	//image_mat_tmp
	image_mat_tmp.resize(THREADS_NUM);
	for(int i=0; i<THREADS_NUM; i++)
	{
		image_mat_tmp[i].resize(n);
		for(int ii=0; ii<n; ii++)
			image_mat_tmp[i][ii].resize(4096);
	}
	
	//image_att_vec
	image_att_vec.resize(entity_num);
	for(int i=0; i<entity_num; i++)
	{
		image_att_vec[i].resize(4096);
		for(int ii=0; ii<4096; ii++)
			image_att_vec[i][ii] = 0;
		for(int ii=0; ii<image_count_vec[i]; ii++)
		{
			//cout << vec_len(image_vec[i][ii]) << endl;
			norm(image_vec[i][ii]);
			for(int iii=0; iii<4096; iii++)
				image_att_vec[i][iii] += image_vec[i][ii][iii];
		}
		norm(image_att_vec[i]);
	}
	
	//image_proj_vec
	image_proj_vec.resize(entity_num);
	for(int i=0; i<entity_num; i++)
		image_proj_vec[i].resize(n);
	
	image_proj_state.resize(entity_num);
	for(int k = 0;k<THREADS_NUM;k++)
	{
		posErrorVec[k].resize(n);
		negErrorVec[k].resize(n);
	}
	
	mut_mutex = PTHREAD_MUTEX_INITIALIZER;
	sgd();		//train with mini-batch SGD
}

void sgd()		//mini-batch SGD
{
	res_triple=0;
	res_ii = 0;
	res_ie = 0;
	res_ei = 0;
	res_normal = 0;
	nbatches=100;		//block number
	batchsize = fb_h.size()/nbatches/THREADS_NUM;		//mini_batch size for each thread
	cout << "batchsize : " << batchsize << endl;
	double step = (rate_begin - rate_end) / (double)nepoch;		//dynamic learning rate step
	cout << step << ' ' << nepoch << endl;
	rate = rate_begin;
	for (int epoch=0; epoch<nepoch; epoch++)
	{
		rate -= step;
		res_triple=0;
		res_ii = 0;
		res_ie = 0;
		res_ei = 0;
		res_normal = 0;
		for (int batch = 0; batch<nbatches; batch++)
		{
			for(int i = 0;i<entity_num;i++)
				image_proj_state[i] = 0;
			for(int k = 0;k<THREADS_NUM;k++)		//init
			{
				res_thread_triple[k] = 0;
				res_thread_ii[k] = 0;
				res_thread_ie[k] = 0;
				res_thread_ei[k] = 0;
				res_thread_normal[k] = 0;
				for (int i=0; i<n; i++)
					for (int ii=0; ii<4096; ii++)
						image_mat_tmp[k][i][ii] = 0;
			}
			relation_tmp = relation_vec;
			entity_tmp = entity_vec;
			//multi-thread
			pthread_t threads[THREADS_NUM];
			for(int k = 0; k < THREADS_NUM; k ++){
				pthread_create(&threads[k], NULL, rand_sel, (void *)k);		//train
			}
			for(int k = 0; k < THREADS_NUM; k ++){
				pthread_join(threads[k], NULL);
			}
			//update
			relation_vec = relation_tmp;
			entity_vec = entity_tmp;
			update_multithread();
			//cout << "update once : " << batch << endl;
		}
		//output
		cout<<"epoch:"<<epoch<<' '<<res_triple<< ' ' << res_ii << ' ' << res_ie << ' ' << res_ei << ' ' << res_normal << endl;
		FILE* f2 = fopen(("../res/relation2vec."+version).c_str(),"w");
		FILE* f3 = fopen(("../res/entity2vec."+version).c_str(),"w");
		FILE* f5 = fopen(("../res/image_mat."+version).c_str(),"w");
		
		for (int i=0; i<relation_num; i++)		//relation_vec
		{
			for (int ii=0; ii<n; ii++)
				fprintf(f2,"%.6lf\t",relation_vec[i][ii]);
			fprintf(f2,"\n");
		}
		for (int i=0; i<entity_num; i++)		//entity_vec
		{
			for (int ii=0; ii<n; ii++)
				fprintf(f3,"%.6lf\t",entity_vec[i][ii]);
			fprintf(f3,"\n");
		}
		for (int i=0; i<n; i++)		//image_mat
		{
			for (int ii=0; ii<4096; ii++)
			{
				fprintf(f5,"%.6lf\t",image_mat[i][ii]);
			}
			fprintf(f5,"\n");
		}
		fclose(f2);
		fclose(f3);
		fclose(f5);
	}
	//overall output
	FILE* f6 = fopen(("../res/image_proj_vec."+version).c_str(),"w");		//image-based representation
	for (int i=0; i<entity_num; i++)
	{
		calc_image_proj_vec(i);
		for (int ii=0; ii<n; ii++)
		{
			fprintf(f6,"%.6lf\t",image_proj_vec[i][ii]);
		}
		fprintf(f6,"\n");
	}
	fclose(f6);
	
}

void *rand_sel(void *tid_void)		//multi-thread
{
	long tid = (long) tid_void;
	for (int k=0; k<batchsize; k++)
	{
		int i=rand_max(fb_h.size());		//positive mark
		int j=rand_max(entity_num);		//negative entity
		double pr = 1000*right_num[fb_r[i]]/(right_num[fb_r[i]]+left_num[fb_r[i]]);
		if (method ==0)
			pr = 500;
		//negative sampling
		
		//||entity_h+r-entity_t||
		int flag_num = rand_max(1000);
		if (flag_num<pr)
		{
			while (ok.count(make_pair(fb_h[i],fb_r[i]))>0&&ok[make_pair(fb_h[i],fb_r[i])].count(j)>0)
				j=rand_max(entity_num);
			train_ee(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i],tid);
		}
		else
		{
			while (ok.count(make_pair(j,fb_r[i]))>0&&ok[make_pair(j,fb_r[i])].count(fb_l[i])>0)
				j=rand_max(entity_num);
			train_ee(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i],tid);
		}
		
		int rel_neg = rand_max(relation_num);		//negative relation
		while (ok.count(make_pair(fb_h[i], rel_neg))>0&& ok[make_pair(fb_h[i], rel_neg)].count(fb_l[i]) > 0)
			rel_neg = rand_max(relation_num);
		train_ee(fb_h[i],fb_l[i],fb_r[i],fb_h[i],fb_l[i],rel_neg,tid);
		
		
		//||image_h+r-image_t||
		if (flag_num<pr)
			train_ii(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i],tid);
		else
			train_ii(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i],tid);
		//train_ii(fb_h[i], fb_l[i], fb_r[i], fb_h[i], fb_l[i],rel_neg,tid);
		
		//||entity_h+r-image_t||
		if (flag_num<pr)
			train_ei(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i],tid);
		else
			train_ei(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i],tid);
		//train_ei(fb_h[i], fb_l[i], fb_r[i], fb_h[i], fb_l[i],rel_neg,tid);
		
		//||image_h+r-entity_t||
		if (flag_num<pr)
			train_ie(fb_h[i],fb_l[i],fb_r[i],fb_h[i],j,fb_r[i],tid);
		else
			train_ie(fb_h[i],fb_l[i],fb_r[i],j,fb_l[i],fb_r[i],tid);
		//train_ie(fb_h[i], fb_l[i], fb_r[i], fb_h[i], fb_l[i],rel_neg,tid);
		
		
		//normalization
		norm(relation_tmp[fb_r[i]]);
		norm(relation_tmp[rel_neg]);
		norm(entity_tmp[fb_h[i]]);
		norm(entity_tmp[fb_l[i]]);
		norm(entity_tmp[j]);
	}
}

void update_multithread()		//multi-thread update
{
	//update
	for(int k = 0;k<THREADS_NUM;k++)
	{
		res_triple += res_thread_triple[k];
		res_ii += res_thread_ii[k];
		res_ie += res_thread_ie[k];
		res_ei += res_thread_ei[k];
		res_normal += res_thread_normal[k];
		for(int i=0; i<n; i++)
		{
			for(int ii=0; ii<4096; ii++)
			{
				image_mat[i][ii] += image_mat_tmp[k][i][ii];
			}
		}
	}
}

//normalization
void gradient_normalization(int ent,int tid,double loss)
{
	for(int i=0; i<n; i++)
	{
		double temp_loss = rate_n*2*image_proj_vec[ent][i];
		for(int ii=0; ii<4096; ii++)
		{
			image_mat_tmp[tid][i][ii] -= temp_loss*image_att_vec[ent][ii];
		}
	}
}

//calc cnn_vec
void calc_image_proj_vec(int ent)		//calculate image-based representation
{
	//calc image_att_vec with attention
	vector<double> attention_vec, temp_image_proj_vec;
	attention_vec.resize(image_count_vec[ent]);
	temp_image_proj_vec.resize(n);
	for(int i=0; i<image_count_vec[ent]; i++)		//calc attention
	{
		for(int ii=0; ii<n; ii++)		//calc proj_vec
		{
			temp_image_proj_vec[ii] = 0;
			for(int iii=0; iii<4096; iii++)
			{
				temp_image_proj_vec[ii] += image_vec[ent][i][iii] * image_mat[ii][iii];
			}
		}
		attention_vec[i] = 0;
		for(int ii=0; ii<n; ii++)		//calc attention
		{
			attention_vec[i] += temp_image_proj_vec[ii] * entity_vec[ent][ii];
		}
		attention_vec[i] = exp(attention_vec[i]);
	}
	double tempExp = 0;
	for(int i=0; i<image_count_vec[ent]; i++)		//attention norm
		tempExp += attention_vec[i];
	for(int i=0; i<image_count_vec[ent]; i++)		//attention norm
		attention_vec[i] /= tempExp;
	//calc image_att_vec
	for(int i=0; i<4096; i++)
		image_att_vec[ent][i] = 0;
	for(int i=0; i<image_count_vec[ent]; i++)
	{
		for(int ii=0; ii<4096; ii++)
		{
			image_att_vec[ent][ii] += attention_vec[i] * image_vec[ent][i][ii];
		}
	}
	//calc image_proc_vec
	for(int i=0; i<n; i++)
	{
		image_proj_vec[ent][i] = 0;
		for(int ii=0; ii<4096; ii++)
		{
			image_proj_vec[ent][i] += image_att_vec[ent][ii] * image_mat[i][ii];
		}
	}
}

double calc_sum_ii(int e1,int e2,int rel, int flag, int tid)		//similarity
{
	double sum=0;
	//calc the projection_vec of image_vec
	if(image_proj_state[e1] == 0)
	{
		calc_image_proj_vec(e1);
		double sum1 = vec_len(image_proj_vec[e1]);
		if(sum1 > 1)
		{
			res_thread_normal[tid] += sum1-1;
			gradient_normalization(e1,tid,sum1-1);
		}
		image_proj_state[e1] = 1;
	}
	if(image_proj_state[e2] == 0)
	{
		calc_image_proj_vec(e2);
		double sum1 = vec_len(image_proj_vec[e2]);
		if(sum1 > 1)
		{
			res_thread_normal[tid] += sum1-1;
			gradient_normalization(e2,tid,sum1-1);
		}
		image_proj_state[e2] = 1;
	}
	
	if(flag == 1)		//positive_sign
	{
		if (L1_flag)
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = image_proj_vec[e2][ii]-image_proj_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)		//calc error
					posErrorVec[tid][ii] = 1;
				else
					posErrorVec[tid][ii] = -1;
			}
		}
		else
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = image_proj_vec[e2][ii]-image_proj_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				posErrorVec[tid][ii] = 2*tempSum;		//calc error
			}
		}
		return sum;
	}
	else		//negative_sign
	{
		if (L1_flag)
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = image_proj_vec[e2][ii]-image_proj_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)		//calc error
					negErrorVec[tid][ii] = 1;
				else
					negErrorVec[tid][ii] = -1;
			}
		}
		else
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = image_proj_vec[e2][ii]-image_proj_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				negErrorVec[tid][ii] = 2*tempSum;		//calc error
			}
		}
		return sum;
	}
}

//head=entity_vec, tail=image_proj_vec
double calc_sum_ei(int e1,int e2,int rel, int flag, int tid)		//similarity
{
	double sum=0;
	if(flag == 1)		//positive_sign
	{
		if (L1_flag)
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = image_proj_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)		//calc error
					posErrorVec[tid][ii] = 1;
				else
					posErrorVec[tid][ii] = -1;
			}
		}
		else
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = image_proj_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				posErrorVec[tid][ii] = 2*tempSum;		//calc error
			}
		}
		return sum;
	}
	else		//negative_sign
	{
		if (L1_flag)
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = image_proj_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)		//calc error
					negErrorVec[tid][ii] = 1;
				else
					negErrorVec[tid][ii] = -1;
			}
		}
		else
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = image_proj_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				negErrorVec[tid][ii] = 2*tempSum;		//calc error
			}
		}
		return sum;
	}
}

//head=image_proj_vec, tail=entity_vec
double calc_sum_ie(int e1,int e2,int rel, int flag, int tid)		//similarity
{
	double sum=0;
	if(flag == 1)		//positive_sign
	{
		if (L1_flag)
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = entity_vec[e2][ii]-image_proj_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)		//calc error
					posErrorVec[tid][ii] = 1;
				else
					posErrorVec[tid][ii] = -1;
			}
		}
		else
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = entity_vec[e2][ii]-image_proj_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				posErrorVec[tid][ii] = 2*tempSum;		//calc error
			}
		}
		return sum;
	}
	else		//negative_sign
	{
		if (L1_flag)
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = entity_vec[e2][ii]-image_proj_vec[e1][ii]-relation_vec[rel][ii];
				sum+=fabs(tempSum);
				if(tempSum > 0)		//calc error
					negErrorVec[tid][ii] = 1;
				else
					negErrorVec[tid][ii] = -1;
			}
		}
		else
		{
			for (int ii=0; ii<n; ii++)
			{
				double tempSum = entity_vec[e2][ii]-image_proj_vec[e1][ii]-relation_vec[rel][ii];
				sum+=sqr(tempSum);
				negErrorVec[tid][ii] = 2*tempSum;		//calc error
			}
		}
		return sum;
	}
}

//head=entity_vec, tail=image_proj_vec
void gradient_ei(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//update
{
	//relation
	for(int i=0; i<n; i++)
	{
		relation_tmp[rel_a][i] += rate_m*posErrorVec[tid][i];
	}
	//head
	for(int i=0; i<n; i++)
	{
		entity_tmp[e1_a][i] += rate_m*posErrorVec[tid][i];
	}
	//tail
	for(int i=0; i<n; i++)
	{
		for(int ii=0; ii<4096; ii++)
		{
			image_mat_tmp[tid][i][ii] -= rate_m*posErrorVec[tid][i]*image_att_vec[e2_a][ii];
		}
	}
	
	//relation
	for (int i = 0;i<n;i++)
	{
		relation_tmp[rel_b][i] -= rate_m*negErrorVec[tid][i];
	}
	//head
	for(int i=0; i<n; i++)
	{
		entity_tmp[e1_b][i] -= rate_m*negErrorVec[tid][i];
	}
	//tail
	for(int i=0; i<n; i++)
	{
		for(int ii=0; ii<4096; ii++)
		{
			image_mat_tmp[tid][i][ii] += rate_m*negErrorVec[tid][i]*image_att_vec[e2_b][ii];
		}
	}
}

//head=image_proj_vec, tail=entity_vec
void gradient_ie(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//update
{
	//relation
	for(int i=0; i<n; i++)
	{
		relation_tmp[rel_a][i] += rate_m*posErrorVec[tid][i];
	}
	//head
	for(int i=0; i<n; i++)
	{
		for(int ii=0; ii<4096; ii++)
		{
			image_mat_tmp[tid][i][ii] += rate_m*posErrorVec[tid][i]*image_att_vec[e1_a][ii];
		}
	}
	//tail
	for(int i=0; i<n; i++)
	{
		entity_tmp[e2_a][i] -= rate_m*posErrorVec[tid][i];
	}
	
	//relation
	for (int i = 0;i<n;i++)
	{
		relation_tmp[rel_b][i] -= rate_m*negErrorVec[tid][i];
	}
	//head
	for(int i=0; i<n; i++)
	{
		for(int ii=0; ii<4096; ii++)
		{
			image_mat_tmp[tid][i][ii] -= rate_m*negErrorVec[tid][i]*image_att_vec[e1_b][ii];
		}
	}
	//tail
	for(int i=0; i<n; i++)
	{
		entity_tmp[e2_b][i] += rate_m*negErrorVec[tid][i];
	}
}

void gradient_ii(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//update
{
	//relation
	for(int i=0; i<n; i++)
	{
		relation_tmp[rel_a][i] += rate_i*posErrorVec[tid][i];
	}
	//head
	for(int i=0; i<n; i++)
	{
		for(int ii=0; ii<4096; ii++)
		{
			image_mat_tmp[tid][i][ii] += rate_i*posErrorVec[tid][i]*image_att_vec[e1_a][ii];
		}
	}
	//tail
	for(int i=0; i<n; i++)
	{
		for(int ii=0; ii<4096; ii++)
		{
			image_mat_tmp[tid][i][ii] -= rate_i*posErrorVec[tid][i]*image_att_vec[e2_a][ii];
		}
	}
	
	//relation
	for (int i = 0;i<n;i++)
	{
		relation_tmp[rel_b][i] -= rate_i*negErrorVec[tid][i];
	}
	//head
	for(int i=0; i<n; i++)
	{
		for(int ii=0; ii<4096; ii++)
		{
			image_mat_tmp[tid][i][ii] -= rate_i*negErrorVec[tid][i]*image_att_vec[e1_b][ii];
		}
	}
	//tail
	for(int i=0; i<n; i++)
	{
		for(int ii=0; ii<4096; ii++)
		{
			image_mat_tmp[tid][i][ii] += rate_i*negErrorVec[tid][i]*image_att_vec[e2_b][ii];
		}
	}
}


//for ||h+r-t||
double calc_sum_ee(int e1,int e2,int rel,int tid)		//similarity
{
	double sum=0;
	if (L1_flag)
		for (int ii=0; ii<n; ii++)
			sum+=fabs(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
	else
		for (int ii=0; ii<n; ii++)
			sum+=sqr(entity_vec[e2][ii]-entity_vec[e1][ii]-relation_vec[rel][ii]);
	return sum;
}

void gradient_ee(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)		//update
{
	for (int ii=0; ii<n; ii++)
	{
		double x = 2*(entity_vec[e2_a][ii]-entity_vec[e1_a][ii]-relation_vec[rel_a][ii]);
		if (L1_flag)
			if (x>0)
				x=1;
			else
				x=-1;
		relation_tmp[rel_a][ii]-=-1*rate*x;
		entity_tmp[e1_a][ii]-=-1*rate*x;
		entity_tmp[e2_a][ii]+=-1*rate*x;
		
		x = 2*(entity_vec[e2_b][ii]-entity_vec[e1_b][ii]-relation_vec[rel_b][ii]);
		if (L1_flag)
			if (x>0)
				x=1;
			else
				x=-1;
		relation_tmp[rel_b][ii]-=rate*x;
		entity_tmp[e1_b][ii]-=rate*x;
		entity_tmp[e2_b][ii]+=rate*x;
	}
}

void train_ee(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)
{
	double sum1 = calc_sum_ee(e1_a,e2_a,rel_a,tid);
	double sum2 = calc_sum_ee(e1_b,e2_b,rel_b,tid);
	if (sum1+margin>sum2)
	{
		res_thread_triple[tid]+=margin+sum1-sum2;
		gradient_ee( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b, tid);
	}
}

void train_ii(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)
{
	double sum1 = calc_sum_ii(e1_a,e2_a,rel_a,1,tid);
	double sum2 = calc_sum_ii(e1_b,e2_b,rel_b,0,tid);
	if (sum1+margin>sum2)
	{
		res_thread_ii[tid]+=margin+sum1-sum2;
		gradient_ii( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b, tid);
	}
}

void train_ie(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)
{
	double sum1 = calc_sum_ie(e1_a,e2_a,rel_a,1,tid);
	double sum2 = calc_sum_ie(e1_b,e2_b,rel_b,0,tid);
	if (sum1+margin>sum2)
	{
		res_thread_ie[tid]+=margin+sum1-sum2;
		gradient_ie( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b, tid);
	}
}

void train_ei(int e1_a,int e2_a,int rel_a,int e1_b,int e2_b,int rel_b,int tid)
{
	double sum1 = calc_sum_ei(e1_a,e2_a,rel_a,1,tid);
	double sum2 = calc_sum_ei(e1_b,e2_b,rel_b,0,tid);
	if (sum1+margin>sum2)
	{
		res_thread_ei[tid]+=margin+sum1-sum2;
		gradient_ei( e1_a, e2_a, rel_a, e1_b, e2_b, rel_b, tid);
	}
}

void add(int x,int y,int z)
{
	fb_h.push_back(x);
	fb_r.push_back(z);
	fb_l.push_back(y);
	ok[make_pair(x,z)][y]=1;
}

void prepare()		//preprocessing
{
    FILE* f1 = fopen("../data/entity2id.txt","r");
	FILE* f2 = fopen("../data/relation2id.txt","r");
	FILE* f3 = fopen("../../image_res/image2count.txt","r");
	FILE* f4 = fopen("../../image_res/image2vec_fc7.txt","r");
	int x;
	double y;
	//build entity2ID¡¢ID2entity map
	while (fscanf(f1,"%s%d",buf,&x)==2)
	{
		string st=buf;
		entity2id[st]=x;		//<entity,ID>
		id2entity[x]=st;		//<ID,entity>
		entity_num++;
	}
	//build relation2ID¡¢ID2relation map
	while (fscanf(f2,"%s%d",buf,&x)==2)
	{
		string st=buf;
		relation2id[st]=x;
		id2relation[x]=st;
		relation_num++;
	}
	//build image2count
	image_count_vec.resize(entity_num);
	while (fscanf(f3,"%s%d",buf,&x)==2)
	{
		string st=buf;
		if(entity2id.count(st) == 1)		//in entity list
			image_count_vec[entity2id[st]] = x;
	}
	
	//build image_feature_vec
	image_vec.resize(entity_num);
	for(int i=0; i<entity_num; i++)
		image_vec[i].resize(image_count_vec[i]);
	vector<int> temp_count_vec;
	temp_count_vec.resize(entity_num);
	for(int i=0; i<entity_num; i++)
		temp_count_vec[i] = 0;
	while (fscanf(f4,"%s",buf)==1)
	{
		string st=buf;
		string temp_ent = st.substr(0, st.find_first_of("_"));
		if(entity2id.count(temp_ent) == 1)		//in entity list
		{
			int temp_ent_id =  entity2id[temp_ent];
			int temp_img_count = temp_count_vec[temp_ent_id];
			image_vec[temp_ent_id][temp_img_count].resize(4096);
			for (int ii=0; ii<4096; ii++)
				fscanf(f4,"%lf",&image_vec[temp_ent_id][temp_img_count][ii]);
			temp_count_vec[temp_ent_id]++;
		}
		else {		//not in list, discard
			for (int ii=0; ii<4096; ii++)
				fscanf(f4,"%lf",&y);
		}
	}
	
	//build triple training set
    FILE* f_kb = fopen("../data/train.txt","r");
	while (fscanf(f_kb,"%s",buf)==1)
    {
        string s1=buf;		//head
        fscanf(f_kb,"%s",buf);
        string s2=buf;		//tail
        fscanf(f_kb,"%s",buf);
        string s3=buf;		//relation
        if (entity2id.count(s1)==0)
        {
            cout<<"miss entity:"<<s1<<endl;
        }
        if (entity2id.count(s2)==0)
        {
            cout<<"miss entity:"<<s2<<endl;
        }
        if (relation2id.count(s3)==0)
        {
            relation2id[s3] = relation_num;
            relation_num++;
        }
        left_entity[relation2id[s3]][entity2id[s1]]++;
        right_entity[relation2id[s3]][entity2id[s2]]++;
        add(entity2id[s1],entity2id[s2],relation2id[s3]);
    }
	
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = left_entity[i].begin(); it!=left_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	left_num[i]=sum2/sum1;
    }
    for (int i=0; i<relation_num; i++)
    {
    	double sum1=0,sum2=0;
    	for (map<int,int>::iterator it = right_entity[i].begin(); it!=right_entity[i].end(); it++)
    	{
    		sum1++;
    		sum2+=it->second;
    	}
    	right_num[i]=sum2/sum1;
    }
    cout<<"relation_num="<<relation_num<<endl;
    cout<<"entity_num="<<entity_num<<endl;
    fclose(f_kb);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc,char**argv)
{
    srand((unsigned) time(NULL));
    int n = 50;		//dimention of entity/relation
	rate = 0.0010;		//learning rate
    rate_begin = 0.0010;		//learning rate parameter
	rate_end = 0.0002;		//learning rate parameter
    double margin = 4;		//loss margin
    int i;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-method", argc, argv)) > 0) method = atoi(argv[i + 1]);
    cout<<"size = "<<n<<endl;
    cout<<"learing rate begin = "<<rate<<endl;
    cout<<"margin = "<<margin<<endl;
    if (method)
        version = "bern";
    else
        version = "unif";
    cout<<"method = "<<version<<endl;
    prepare();		//preprocessing & training
    run(n,rate,margin,method);
}
