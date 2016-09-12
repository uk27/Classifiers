#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <cassert>
#include <math.h>
#include <stdio.h>
#include <map>
#include <cstdio>
#include <time.h>
#include <iomanip>
#include <algorithm> 
#include <cstring>
#include<string>
#include <sstream>
using namespace std;

int ob_ptr=0;
int dim_ptr=0;
int max_dim=0;
int count1 =0;
float f1_measure=0;
float lambda=1;

int n_clusters=20;
int k_val=5;
float **centroid_1;
float **centroid_2;
float **sim_mat;

string input_file="";
string class_file="";
string train_file="";
string test_file="";
string rlabel="";
string feature="";
string outputfile="";
string representation="";

std::map<string, int> mapping1;
std::vector<string> mapping2;

std::vector<int> class_info;
std::map<int, vector<int> > class_map;
std::vector<int> train_data;
std::vector<int> test_data;
std::map<int, std::vector<int> > test_map;
std::vector<float> f1_score;

std:: vector<int> class_result;
std:: map<int, std::vector<int> > pred_map;

std::vector<int> doc_freq;


namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}
//To store 1 if a file is in the training data
int train_present[6683]={0};

struct sort_desc2 {
    bool operator()(const std::pair<int,float> &left, const std::pair<int,float> &right) {
        return left.second > right.second;
    }
    };

void read_trainfile()
{
	string ID1;
	int id=0;
	ifstream fc;
	fc.open(train_file.c_str());
	while(std::getline(fc, ID1))
	{
		id=atoi(ID1.c_str());
		
		train_data.push_back(id);
		train_present[id-1]++;
		
	}
        
	//train_data.pop_back();
	
	fc.close();
	//for (int i=0; i<train_data.size();i++)
	//cout<<train_data[i]<<"\n";
}
		
void read_testfile()
{
	string ID1;
	ifstream fc;
	int c=0, d=0;
	fc.open(test_file.c_str());
	while(std::getline(fc, ID1))
	{
//		std::getline(fc, ID1);
		d=atoi(ID1.c_str());
		test_data.push_back(d);
		c= class_info[d-1]-1;
		test_map[c].push_back(d);
		//cout<<"Test File: "<< d <<" Class no: " << c+1 << endl;
		
	}
		
        
	//test_data.pop_back();

	/*for (std:: map<int, std::vector<int> >::iterator it =test_map.begin(); it!=test_map.end(); ++it)
	{
		cout<<"class"<<it->first<<" ";
		for (int d=0; d< it->second.size(); d++)
		{
			cout<<it->second[d] << " ";
		}
	}*/
	//cout<<endl<<endl;
	fc.close();
	//for (int i=0; i<test_data.size();i++)
	//cout<<test_data[i]<<"\n";
}

void writeToFile(){

  ofstream myfile;
  myfile.open (outputfile.c_str());
  string s="";

  for (int i=0;  i<class_result.size(); i++)
	{
		s+= patch::to_string(test_data[i]);
		s=s+" ";
		s=s+ patch::to_string(mapping2[class_result[i]]) ;
		s=s+ "\n";
		
	}
  //string x= patch::to_string(1);
  myfile << s << endl;
  myfile.close();

}

void read_classfile()
{
		string ID1, ID2;
		
		int count1 = 0;
		ifstream fc;
		fc.open(class_file.c_str());
		while(fc)
		{
				count1++;

				std::getline(fc, ID1, ' ');
				std::getline(fc, ID2);
				//cout<<ID1<< " " << ID2<<endl;
				if ( mapping1.find(ID2) == mapping1.end() ) {
						mapping1[ID2]= mapping1.size();
					mapping2.push_back(ID2);
				}

				class_info.push_back(mapping1[ID2]);
				class_map[mapping1[ID2]].push_back(atoi(ID1.c_str()));
				
				

		}
		
		
		class_info.pop_back();
		class_map[mapping1[ID2]].pop_back();

		 //Print the class
		//for (int it=0; it< mapping2.size(); it++)
		//	cout<<mapping2[it]<<endl;
			//cout<<"Document:"<<i+1<< " class: " << class_info[i]<<"\n";
		
}		


void centroid_initializer(int *ob, int *dim, float *val)
{

		centroid_1 = (float **)calloc(n_clusters+1,sizeof(float*));
		centroid_2 = (float **)calloc(n_clusters+1,sizeof(float*));

		for(int t=0;t<n_clusters;t++)
			{
				centroid_1[t] = (float*)calloc(max_dim+1,sizeof(float));
				centroid_2[t] = (float*)calloc(max_dim+1,sizeof(float));
			}

		int j=0, beg=0, end1=0, doc=0, c=0;
		
		
		for(int d=0; d<train_data.size();d++)
		{
				doc= train_data[d]-1; //Since document no d will be stored in d-1 th loc
				c= class_info[doc]-1; //Since class number starts with 1

				beg= (doc==0 ? 0 : ob[doc-1]+ 1);
				end1= ob[doc];
                     
                                for(int i=0; i<n_clusters; i++)
				{
					j=beg;
					
					while(j<=end1)
					{
						i==c ? (centroid_1[i][dim[j]-1]+= val[j]) : (centroid_2[i][dim[j]-1]+= val[j]);
						//cout<<centroid_1[i][dim[j]-1]<<"  "<<centroid_2[i][dim[j]-1]<<endl;
						j++;
					}

				}
		}

}

void print_classres()
{
	int count=0;
	
	for (int i=0;  i<class_result.size(); i++)
	{
		//cout<<"Document: " << test_data[i] << "Class: " << class_result[i] +1 << endl ;
		cout<<"Document: " << test_data[i] << "Class: " << mapping2[class_result[i]] ;
		if((class_result[i] +1) == class_info[test_data[i]-1])
			count++;
	}
		cout<<"\n No of matches= "<<count;
		//cout<<"\n Fraction matched= "<<float(count/test_data.size());
}

void centroid_normalizer ()
{
		float mod_1[n_clusters];
		float mod_2[n_clusters];

		//float modi=0;
		for(int k=0;k<n_clusters;k++)
		{
				mod_1[k]=0;
				mod_2[k]=0;
				for (int j=0; j<max_dim;j++)
				{
						mod_1[k]+= pow(centroid_1[k][j],2);
						mod_2[k]+= pow(centroid_2[k][j],2);
				}

				mod_1[k]= pow(mod_1[k], 0.5);
				mod_2[k]= pow(mod_2[k], 0.5);
		}


		for(int k=0;k<n_clusters;k++)
		{

				for(int j=0;j<max_dim;j++)
				{
						centroid_1[k][j]= centroid_1[k][j]/mod_1[k];
						centroid_2[k][j]= centroid_2[k][j]/mod_2[k];
				}

		}
}

void print_centroid()
{
	for(int k=0;k<n_clusters;k++)
		{

				for(int j=0;j<3;j++)
				{
						cout<< "c" << k+1 <<":" << centroid_1[k][j];
						cout<< "  c'" << k+1 <<":" << centroid_2[k][j] << " ";
				}
				cout<<endl;

		}
}

void centroid_classifier(int *ob, int *dim, float *val)
{

 	float max_d=0, v=0;
	int doc=0;
	int beg=0, end1=0, j=0;
	float sim1=0, sim2=0;

	//Create a cXd matrix (sim_mat) to store score between cth class with dth doc in test_data
	
		sim_mat = (float **)calloc(n_clusters+1,sizeof(float*));

		for(int t=0;t<n_clusters;t++)
			{
				sim_mat[t] = (float*)calloc(test_data.size()+1,sizeof(float));
				
			}

	
	for (int d=0; d< test_data.size(); d++)
	{
		doc= test_data[d]-1;
		max_d=-1000;
		beg= (doc==0 ? 0 : ob[doc-1]+1);
		end1 = ob[doc];
		//cout<<endl<<doc;
		for (int c=0; c < n_clusters; c++)
		{
			v=0; 
			sim1=0, sim2=0;
			j=beg;
                        //cout<<c<< " " ;
			while (j<=end1)
				
				{
					sim1+= centroid_1[c][dim[j]-1]*val[j];
					sim2+= centroid_2[c][dim[j]-1]*val[j];
					j++;					
				}

			//val= sim_compute(doc, class_info[c]-1);

			v= sim1- sim2;
			sim_mat[c][d]=v;
			//cout<< v << " ";				
			
			if(v> max_d)
			{	
				max_d= v;
				class_result[d]= c;
				pred_map[c].push_back(test_data[d]);
				
			}
		}
	}
}


void cal_f1()
{
	
	int tp=0, fn=0, fp=0;
	float max_f1=0, this_f1=0;
	std::vector <pair <int, float> > vec;

	for(int c=0; c< n_clusters; c++)
	{
		max_f1=0;
		for(int d=0; d< test_data.size(); d++)
		{
			vec.push_back(make_pair(train_data[d], sim_mat[c][d]));
		}
		    std::sort(vec.begin(), vec.end(), sort_desc2());

		//for(int d=0; d< test_data.size(); d++)
		//	cout<<vec[d].first<<endl;

		tp=0;
		fp=0;		
		fn= test_map[c].size();
	
		
		for(int d=0; d< test_data.size(); d++)
		{
			
			if (class_info[vec[d].first]-1 == c)
			{
				tp++; 
				fn--;
			}
			else 
				{fp++;}

			this_f1= 2*tp/float(2*tp + fp + fn);
			//cout<<"tp= "<<tp<<" fp="<<fp<<" fn="<<fn<< " thisf1="<< this_f1<< endl;
			//cout<<this_f1<<endl;
			
			if(this_f1> max_f1)
				max_f1= this_f1;
		}
		
		f1_score.push_back(max_f1);
		vec.resize(0);
		

	}




}


int main(int c, char** argv)
{

		input_file=argv[1];
		class_file=argv[2];
		train_file=argv[3];
		test_file=argv[4];	

		class_file=argv[5];	
		feature=argv[6];	
		representation=argv[7];	
		outputfile=argv[8];	
		
		string r1= "tf";
		string r2= "binary";
		string r3= "sqrt";
		string r4= "tfidf";
		string r5= "binaryidf";
		string r6= "sqrtidf";
		//representation= argv[5];
		
		
		vector<int> o;
		vector<int> dm;
		vector<float> vm;
		ifstream f1,f2;
		string ID;

	        read_classfile();
		read_trainfile();
		read_testfile();

		f1.open(input_file.c_str());
		while(f1)
		{
				count1++;
				std::getline(f1, ID, ' ');
				o.push_back(atoi(ID.c_str()));
				std::getline(f1, ID, ' ');
				dm.push_back(atoi(ID.c_str()));
				std::getline(f1, ID);
				vm.push_back(atoi(ID.c_str()));

		}

		--count1;
		//cout<<"Total number of non- zero dimensions in the entire data set: "<< count1 << endl;
		//The above is because it stores a line extra. So at count1-1= total number of lines.
		//It is stored in o from 0 to (count1-1)-1 th location = count1-2
		//cout<<count1;

		int x= o[count1-1]; //x is the object stored in the last line
		//cout<<"x="<<x;
		int ob[x];

		int *d = (int*)calloc(count1+1,sizeof(int));
		float *v = (float*)calloc(count1+1,sizeof(float));

		int prev= o[0];
		int cur= prev;
		int i;


		for(i=0; i<=count1; i++) //o.size = total no if lines including one extra blank line
		{
				cur= o[i];

				d[i]=dm[i];

				
				//Taking max_dim
				if(d[i]>max_dim)
						max_dim=d[i];

				v[i]=vm[i];

				if(cur!=prev)
				{
						ob[ob_ptr++]= i-1;
						prev=cur;
						//cout<<ob_ptr<<endl;


				}


		}


		dim_ptr= i-2;
		ob[ob_ptr]= dim_ptr; //Same as storing count1 -2
		//cout<<endl<<"dim_ptr="<<dim_ptr;

		//dim_ptr=i-2;
		//cout<<"dim_ptr="<<dim_ptr<<endl;
		//cout<<"Number of Objects= "<<ob_ptr;
		//cout<<"Number of distinct dimensions= "<<max_dim <<endl;
		vector<int>().swap(o);
		//Initialize doc_frequency
		for (int df=0; df< max_dim; df++)
			{
				doc_freq.push_back(1);
			}
		

		int j=0;
		float t_mod=0;
		float mod[x];

		int jj=0;
		for(int i=0; i<ob_ptr; i++)
			{
				while(jj<= ob[i]){
				//If i is in training file, update doc_freq
				if (train_present[i])
					doc_freq[d[jj]-1]++;
				
				 jj++;
				}
			}	

		
		//Calculating mod
		for(int i=0;i<ob_ptr;i++)
		{
				t_mod=0;

				while(j<= ob[i])
				{

						if(strcmp(representation.c_str(), r2.c_str())==0)
							v[j]=1;

						else if(strcmp(representation.c_str(), r3.c_str())==0)	
							v[j]= pow(v[j], 0.5);

						else if(strcmp(representation.c_str(), r4.c_str())==0)
							v[j]= v[j]* log(train_data.size()/doc_freq[d[j]-1]);

						else if(strcmp(representation.c_str(), r5.c_str())==0)
							v[j]= doc_freq[d[j]-1];
				
						else if(strcmp(representation.c_str(), r6.c_str())==0)
							v[j]= pow(v[j], 0.5)*log(train_data.size()/doc_freq[d[j]-1]);
						
						//cout<<j+1<<" "<<i+1<<" "<<d[j]<<" "<<v[j]<<endl;
						t_mod+= pow(v[j],2);
						j++;
		
				}

				mod[i]= pow(t_mod,0.5);
		}


		//Dividing by mod
		float s=0;
		j=0;
		for(int i=0;i<ob_ptr;i++)
		{

				while(j<= ob[i])
				{
						v[j]=v[j]/mod[i];
					
						

						j++;
	
						
						

				}

		}

                vector<int> init_to_minus1(test_data.size(), -1);
		class_result= init_to_minus1;

		centroid_initializer(ob, d, v);
		centroid_normalizer ();
		//print_centroid();
		centroid_classifier(ob, d, v);
		//print_classres();
		writeToFile();
		
		cal_f1();
		
		for (int c=0; c< n_clusters; c++)
			cout<< f1_score[c]<<endl;
		cout<<"Executed Successfully"; 
		
		return 0;

}




