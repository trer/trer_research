/* ******************************************************************************
 * newscan.cpp
 * 
 * parsing algorithm for bwt construction of repetitive sequences based 
 * on prefix free parsing. See:
 *   Christina Boucher, Travis Gagie, Alan Kuhnle and Giovanni Manzini
 *   Prefix-Free Parsing for Building Big BWTs
 *   [Proc. WABI '18](http://drops.dagstuhl.de/opus/volltexte/2018/9304/)
 * 
 * Usage:
 *   newscan.x wsize modulus file
 * 
 * Unless the parameter -c (compression rather than BWT construction,
 * see "Compression mode" below) the input file cannot contain 
 * the characters 0x0, 0x1, 0x2 which are used internally. 
 * 
 * Since the i-th thread accesses the i-th segment of the input file 
 * random access (fseek) must be possible. For gzipped inputs use 
 * cnewscan.x which doesn't use threads but automatically extracts the 
 * content from a gzipped input using the lz library. 
 * 
 * The parameters wsize and modulus are used to define the prefix free parsing 
 * using KR-fingerprints (see paper)
 * 
 * 
 * *** BWT construction ***
 *  
 * The algorithm computes the prefix free parsing of 
 *     T = (0x2)file_content(0x2)^wsize
 * in a dictionary of words D and a parsing P of T in terms of the  
 * dictionary words. Note that the words in the parsing overlap by wsize.
 * Let d denote the number of words in D and p the number of phrases in 
 * the parsing P
 * 
 * newscan.x outputs the following files:
 * 
 * file.dict
 * containing the dictionary words in lexicographic order with a 0x1 at the end of
 * each word and a 0x0 at the end of the file. Size: |D| + d + 1 where
 * |D| is the sum of the word lengths
 * 
 * file.occ
 * the number of occurrences of each word in lexicographic order.
 * We assume the number of occurrences of each word is at most 2^32-1
 * so the size is 4d bytes
 * 
 * file.parse
 * containing the parse P with each word identified with its 1-based lexicographic 
 * rank (ie its position in D). We assume the number of distinct words
 * is at most 2^32-1, so the size is 4p bytes
 * 
 * file.last 
 * containing the character in position w+1 from the end for each dictionary word
 * Size: p
 * 
 * file.sai (if option -s is given on the command line) 
 * containing the ending position +1 of each parsed word in the original
 * text written using IBYTES bytes for each entry (IBYTES defined in utils.h) 
 * Size: p*IBYTES
 * 
 * The output of newscan.x must be processed by bwtparse, which invoked as
 * 
 *    bwtparse file
 * 
 * computes the BWT of file.parse and produces file.ilist of size 4p+4 bytes
 * contaning, for each dictionary word in lexicographic order, the list 
 * of BWT positions where that word appears (ie i\in ilist(w) <=> BWT[i]=w).
 * There is also an entry for the EOF word which is not in the dictionary 
 * but is assumed to be the smallest word.  
 * 
 * In addition, bwtparse permutes file.last according to
 * the BWT permutation and generates file.bwlast such that file.bwlast[i] 
 * is the char from P[SA[i]-2] (if SA[i]==0 , BWT[i]=0 and file.bwlast[i]=0, 
 * if SA[i]==1, BWT[i]=P[0] and file.bwlast[i] is taken from P[n-1], the last 
 * word in the parsing).  
 * 
 * If the option -s is given to bwtparse, it permutes file.sai according
 * to the BWT permutation and generate file.bwsai using again IBYTES
 * per entry.  file.bwsai[i] is the ending position+1 of BWT[i] in the 
 * original text 
 * 
 * The output of bwtparse (the files .ilist .bwlast) together with the
 * dictionary itself (file .dict) and the number of occurrences
 * of each word (file .occ) are used to compute the final BWT by the 
 * pfbwt algorithm.
 *
 * As an additional check to the correctness of the parsing, it is 
 * possible to reconstruct the original file from the files .dict
 * and .parse using the unparse tool.  
 * 
 * 
 *  *** Compression mode ***
 * 
 * If the -c option is used, the parsing is computed for compression
 * purposes rather than for building the BWT. In this case the redundant 
 * information (phrases overlaps and 0x2's) is not written to the output files.
 * 
 * In addition, the input can contain also the characters 0x0, 0x1, 0x2
 * (ie can be any input file). The program computes a quasi prefix-free 
 * parsing (with no overlaps): 
 * 
 *   T = w_0 w_1 w_2 ... w_{p-1}
 *
 * where each word w_i, except the last one, ends with a lenght-w suffix s_i
 * such that KR(s_i) mod p = 0 and s_i is the only lenght-w substring of
 * w_i with that property, with the possible exception of the lenght-w
 * prefix of w_0.
 * 
 * In Compression mode newscan.x outputs the following files:
 * 
 * file.dicz
 * containing the concatenation of the (distinct) dictionary words in 
 * lexicographic order. 
 * Size: |D| where |D| is the sum of the word lengths
 * 
 * file.dicz.len
 * containing the lenght in bytes of the dictionary words again in 
 * lexicographic order. Each lenght is represented by a 32 bit int.
 * Size: 4d where d is the number of distinct dictionary words. 
 * 
 * file.parse
 * containing the parse P with each word identified with its 1-based lexicographic 
 * rank (ie its position in D). We assume the number of distinct words
 * is at most 2^32-1, so the size is 4p bytes.
 * 
 * From the above three files it is possible to recover the original input
 * using the unparsz tool.
 * 
 */
#include <assert.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/stat.h>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>
#include <vector>
#include <map>
#ifdef GZSTREAM
#include <gzstream.h>
#endif
extern "C" {
#include "utils.h"
}

using namespace std;
using namespace __gnu_cxx;



// =============== algorithm limits =================== 
// maximum number of distinct words
#define MAX_DISTINCT_WORDS (INT32_MAX -1)
typedef uint32_t word_int_t;
// maximum number of occurrences of a single word
#define MAX_WORD_OCC (UINT32_MAX)
typedef uint32_t occ_int_t;

// values of the wordFreq map: word, its number of occurrences, and its rank
struct word_stats {
  string str;
  occ_int_t occ;
  word_int_t rank=0;
};

// -------------------------------------------------------------
// struct containing command line parameters and other globals
struct Args {
   string inputFileName = "";
   int w = 10;            // sliding window size and its default 
   int p = 100;           // modulus for establishing stopping w-tuples 
   bool SAinfo = false;   // compute SA information
   bool compress = false; // parsing called in compress mode 
   int th=0;              // number of helper threads
   int verbose=0;         // verbosity level 
   bool reverse=0;        // direction of file read
};


// -----------------------------------------------------------------
// class to maintain a window in a string and its KR fingerprint
struct KR_window {
  int wsize;
  int *window;
  int asize;
  const uint64_t prime = 1999999973;
  uint64_t hash;
  uint64_t tot_char;
  uint64_t asize_pot;   // asize^(wsize-1) mod prime 
  
  KR_window(int w): wsize(w) {
    asize = 256;
    asize_pot = 1;
    for(int i=1;i<wsize;i++) 
      asize_pot = (asize_pot*asize)% prime; // ugly linear-time power algorithm  
    // alloc and clear window
    window = new int[wsize];
    reset();     
  }
  
  // init window, hash, and tot_char 
  void reset() {
    for(int i=0;i<wsize;i++) window[i]=0;
    // init hash value and related values
    hash=tot_char=0;    
  }
  
  uint64_t addchar(int c) {
    int k = tot_char++ % wsize;
    // complex expression to avoid negative numbers 
    hash += (prime - (window[k]*asize_pot) % prime); // remove window[k] contribution  
    hash = (asize*hash + c) % prime;      //  add char i 
    window[k]=c;
    // cerr << get_window() << " ~~ " << window << " --> " << hash << endl;
    return hash; 
  }
  // debug only 
  string get_window() {
    string w = "";
    int k = (tot_char-1) % wsize;
    for(int i=k+1;i<k+1+wsize;i++)
      w.append(1,window[i%wsize]);
    return w;
  }
  
  ~KR_window() {
    delete[] window;
  } 

};
// -----------------------------------------------------------


// -----------------------------------------------------------------
// class to read file backwards efficiently
struct backward_ifstream {
  static const long buff_size = 4096;
  int buff[buff_size];
  long cur;

  ifstream f;
  long pos;

  backward_ifstream(ifstream&& ifs) {
    f = std::move(ifs);

    pos = f.tellg();
    if (pos < buff_size - 1) {
      f.seekg(0);
      for (long p = 0; p <= pos; ++p) {
        buff[p] = f.get();
      }
      cur = pos;
    } else {
      f.seekg(-(buff_size-1),ios::cur);
      for (long p = 0; p < buff_size; ++p) {
        buff[p] = f.get();
      }
      cur = buff_size-1;
    }
  }

  ~backward_ifstream() { f.close(); }

  void close() { f.close(); }

  int get() {
    if (pos < 0) return EOF;

    int res = buff[cur]; pos--;

    if (cur == 0) {
      if (pos < buff_size - 1) {
        f.seekg(0);
        for (long p = 0; p <= pos; ++p) {
          buff[p] = f.get();
        }
        cur = pos;
      } else {
        f.seekg(pos-(buff_size-1));
        for (long p = 0; p < buff_size; ++p) {
          buff[p] = f.get();
        }
        cur = buff_size-1;
      }
    } else {
      cur--;
    }
    return res;
  }
};


static void save_update_word(Args& arg, string& w, map<uint64_t,word_stats>&  freq, FILE *tmp_parse_file, FILE *last, FILE *sa, uint64_t &pos);

#ifndef NOTHREADS
#include "newscan.hpp"
#endif



// compute 64-bit KR hash of a string 
// to avoid overflows in 64 bit aritmethic the prime is taken < 2**55
uint64_t kr_hash(string s) {
    uint64_t hash = 0;
    //const uint64_t prime = 3355443229;     // next prime(2**31+2**30+2**27)
    const uint64_t prime = 27162335252586509; // next prime (2**54 + 2**53 + 2**47 + 2**13)
    for(size_t k=0;k<s.size();k++) {
      int c = (unsigned char) s[k];
      assert(c>=0 && c< 256);
      hash = (256*hash + c) % prime;    //  add char k
    } 
    return hash; 
}



// save current word in the freq map and update it leaving only the 
// last minsize chars which is the overlap with next word  
static void save_update_word(Args& arg, string& w, map<uint64_t,word_stats>& freq, FILE *tmp_parse_file, FILE *last, FILE *sa, uint64_t &pos)
{
  size_t minsize = arg.w; 
  assert(pos==0 || w.size() > minsize);
  if(w.size() <= minsize) return;
  // save overlap 
  string overlap(w.substr(w.size() - minsize)); // keep last minsize chars
  // if we are compressing, discard the overlap
  if(arg.compress)
     w.erase(w.size() - minsize); // erase last minsize chars 
  
  // get the hash value and write it to the temporary parse file
  uint64_t hash = kr_hash(w);
  if(fwrite(&hash,sizeof(hash),1,tmp_parse_file)!=1) die("parse write error");

#ifndef NOTHREADS
  xpthread_mutex_lock(&map_mutex,__LINE__,__FILE__);
#endif  
  // update frequency table for current hash
  if(freq.find(hash)==freq.end()) {
      freq[hash].occ = 1; // new hash
      freq[hash].str = w; 
  }
  else {
      freq[hash].occ += 1; // known hash
      if(freq[hash].occ <=0) {
        cerr << "Emergency exit! Maximum # of occurence of dictionary word (";
        cerr<< MAX_WORD_OCC << ") exceeded\n";
        exit(1);
      }
      if(freq[hash].str != w) {
        cerr << "Emergency exit! Hash collision for strings:\n";
        cerr << freq[hash].str << "\n  vs\n" <<  w << endl;
        exit(1);
      }
  }
#ifndef NOTHREADS
  xpthread_mutex_unlock(&map_mutex,__LINE__,__FILE__);
#endif
  if(arg.compress) 
    pos += w.size(); // if compressing, just update position 
  else {
    // update last/sa files  
    // output char w+1 from the end
    if(fputc(w[w.size()- minsize-1],last)==EOF) die("Error writing to .last file");
    // compute ending position +1 of current word and write it to sa file 
    // pos is the ending position+1 of the previous word and is updated here 
    if(pos==0) pos = w.size()-1; // -1 is for the initial $ of the first word
    else pos += w.size() -minsize; 
    if(sa) if(fwrite(&pos,IBYTES,1,sa)!=1) die("Error writing to sa info file");
  } 
  // keep only the overlapping part of the window
  w.assign(overlap);
}



// prefix free parse of file fnam. w is the window size, p is the modulus 
// use a KR-hash as the word ID that is immediately written to the parse file
uint64_t process_file(Args& arg, map<uint64_t,word_stats>& wordFreq)
{
  //open a, possibly compressed, input file
  string fnam = arg.inputFileName;
  #ifdef GZSTREAM 
  igzstream f(fnam.c_str());
  #else
  ifstream f(fnam);
  #endif    
  if(!f.rdbuf()->is_open()) {// is_open does not work on igzstreams 
    perror(__func__);
    throw new std::runtime_error("Cannot open input file " + fnam);
  }

  // open the 1st pass parsing file 
  FILE *g = open_aux_file(arg.inputFileName.c_str(),EXTPARS0,"wb");
  FILE *sa_file = NULL, *last_file=NULL;
  if(!arg.compress) {
    // open output file containing the char at position -(w+1) of each word
    last_file = open_aux_file(arg.inputFileName.c_str(),EXTLST,"wb");  
    // if requested open file containing the ending position+1 of each word
    if(arg.SAinfo) 
      sa_file = open_aux_file(arg.inputFileName.c_str(),EXTSAI,"wb");
  }
  
  
  // main loop on the chars of the input file
  int c;
  uint64_t pos = 0; // ending position +1 of previous word in the original text, used for computing sa_info 
  assert(IBYTES<=sizeof(pos)); // IBYTES bytes of pos are written to the sa info file 
  // init first word in the parsing with a Dollar char unless we are just compressing
  string word("");
  if(!arg.compress) word.append(1,Dollar);
  // init empty KR window: constructor only needs window size
  KR_window krw(arg.w);
  while( (c = f.get()) != EOF ) {
    if(c<=Dollar && !arg.compress) {
      // if we are not simply compressing then we cannot accept 0,1,or 2
      cerr << "Invalid char found in input file. Exiting...\n"; exit(1);
    }
    word.append(1,c);
    uint64_t hash = krw.addchar(c);
    if(hash%arg.p==0) {
      // end of word, save it and write its full hash to the output file
      // cerr << "~"<< c << "~ " << hash << " ~~ <" << word << "> ~~ <" << krw.get_window() << ">" <<  endl;
      save_update_word(arg,word,wordFreq,g,last_file,sa_file,pos);
    }    
  }
  // virtually add w null chars at the end of the file and add the last word in the dict
  word.append(arg.w,Dollar);
  save_update_word(arg,word,wordFreq,g,last_file,sa_file,pos);
  // close input and output files 
  if(sa_file) if(fclose(sa_file)!=0) die("Error closing SA file");
  if(last_file) if(fclose(last_file)!=0) die("Error closing last file");  
  if(fclose(g)!=0) die("Error closing parse file");
  if(arg.compress)
    assert(pos==krw.tot_char);
  else 
    assert(pos==krw.tot_char+arg.w);
  // if(pos!=krw.tot_char+arg.w) cerr << "Pos: " << pos << " tot " << krw.tot_char << endl;
  f.close();
  return krw.tot_char;
}

uint64_t process_file_reverse(Args& arg, map<uint64_t,word_stats>& wordFreq)
{
  //open a, possibly compressed, input file
  string fnam = arg.inputFileName;
  #ifdef GZSTREAM 
  igzstream f(fnam.c_str(), std::ifstream::ate);
  #else
  ifstream f(fnam, std::ifstream::ate);
  #endif    
  if(!f.rdbuf()->is_open()) {// is_open does not work on igzstreams 
    perror(__func__);
    throw new std::runtime_error("Cannot open input file " + fnam);
  }

  long size = f.tellg();
  f.seekg(size-1);

  string revFileName = arg.inputFileName + ".rev";

  // open the 1st pass parsing file 
  FILE *g = open_aux_file(revFileName.c_str(),EXTPARS0,"wb");
  FILE *sa_file = NULL, *last_file=NULL;
  if(!arg.compress) {
    // open output file containing the char at position -(w+1) of each word
    last_file = open_aux_file(revFileName.c_str(),EXTLST,"wb");  
    // if requested open file containing the ending position+1 of each word
    if(arg.SAinfo) 
      sa_file = open_aux_file(revFileName.c_str(),EXTSAI,"wb");
  }
  
  
  // main loop on the chars of the input file
  backward_ifstream bf(std::move(f));
  int c;
  uint64_t pos = 0; // ending position +1 of previous word in the original text, used for computing sa_info
  assert(IBYTES<=sizeof(pos)); // IBYTES bytes of pos are written to the sa info file 
  // init first word in the parsing with a Dollar char unless we are just compressing
  string word("");
  if(!arg.compress) word.append(1,Dollar);
  // init empty KR window: constructor only needs window size
  KR_window krw(arg.w);
  // while( (c = f.get()) != EOF ) {
  while((c=bf.get()) != EOF) {
    if(c<=Dollar && !arg.compress) {
      // if we are not simply compressing then we cannot accept 0,1,or 2
      cerr << "Invalid char found in input file. Exiting...\n"; exit(1);
    }
    word.append(1,c);
    uint64_t hash = krw.addchar(c);
    if(hash%arg.p==0) {
      // end of word, save it and write its full hash to the output file
      // cerr << "~"<< c << "~ " << hash << " ~~ <" << word << "> ~~ <" << krw.get_window() << ">" <<  endl;
      save_update_word(arg,word,wordFreq,g,last_file,sa_file,pos);
    }  
  }
  // virtually add w null chars at the end of the file and add the last word in the dict
  word.append(arg.w,Dollar);
  save_update_word(arg,word,wordFreq,g,last_file,sa_file,pos);
  // close input and output files 
  if(sa_file) if(fclose(sa_file)!=0) die("Error closing SA file");
  if(last_file) if(fclose(last_file)!=0) die("Error closing last file");  
  if(fclose(g)!=0) die("Error closing parse file");
  if(arg.compress)
    assert(pos==krw.tot_char);
  else 
    assert(pos==krw.tot_char+arg.w);
  // if(pos!=krw.tot_char+arg.w) cerr << "Pos: " << pos << " tot " << krw.tot_char << endl;
  bf.close();
  return krw.tot_char;
}

// function used to compare two string pointers
bool pstringCompare(const string *a, const string *b)
{
  return *a <= *b;
}

// given the sorted dictionary and the frequency map write the dictionary and occ files
// also compute the 1-based rank for each hash
void writeDictOcc(Args &arg, map<uint64_t,word_stats> &wfreq, vector<const string *> &sortedDict)
{
  assert(sortedDict.size() == wfreq.size());
  FILE *fdict, *fwlen=NULL, *focc=NULL;
  // open dictionary and occ files
  if(arg.compress) {
    fdict = open_aux_file(arg.inputFileName.c_str(),EXTDICZ,"wb");
    fwlen = open_aux_file(arg.inputFileName.c_str(),EXTDZLEN,"wb");
  }
  else {
    fdict = open_aux_file(arg.inputFileName.c_str(),EXTDICT,"wb");
    focc = open_aux_file(arg.inputFileName.c_str(),EXTOCC,"wb");
  }
  
  word_int_t wrank = 1; // current word rank (1 based)
  for(auto x: sortedDict) {          // *x is the string representing the dictionary word
    const char *word = (*x).data();       // current dictionary word
    size_t len = (*x).size();  // offset and length of word
    assert(len>(size_t)arg.w || arg.compress);
    uint64_t hash = kr_hash(*x);
    auto& wf = wfreq.at(hash);
    assert(wf.occ>0);
    size_t s = fwrite(word,1,len, fdict);
    if(s!=len) die("Error writing to DICT file");
    if(arg.compress) {
      s = fwrite(&len,4,1,fwlen);
      if(s!=1) die("Error writing to WLEN file");
    }
    else {
      if(fputc(EndOfWord,fdict)==EOF) die("Error writing EndOfWord to DICT file");
      s = fwrite(&wf.occ,sizeof(wf.occ),1, focc);
      if(s!=1) die("Error writing to OCC file");
    }
    assert(wf.rank==0);
    wf.rank = wrank++;
  }
  if(arg.compress) {
    if(fclose(fwlen)!=0) die("Error closing WLEN file");
  }
  else {
    if(fputc(EndOfDict,fdict)==EOF) die("Error writing EndOfDict to DICT file");
    if(fclose(focc)!=0) die("Error closing OCC file");
  }
  if(fclose(fdict)!=0) die("Error closing DICT file");
}

void remapParse(Args &arg, map<uint64_t,word_stats> &wfreq)
{
  // open parse files. the old parse can be stored in a single file or in multiple files
  mFile *moldp = mopen_aux_file(arg.inputFileName.c_str(), EXTPARS0, arg.th);
  FILE *newp = open_aux_file(arg.inputFileName.c_str(), EXTPARSE, "wb");

  // recompute occ as an extra check 
  vector<occ_int_t> occ(wfreq.size()+1,0); // ranks are zero based 
  uint64_t hash;
  while(true) {
    size_t s = mfread(&hash,sizeof(hash),1,moldp);
    if(s==0) break;
    if(s!=1) die("Unexpected parse EOF");
    word_int_t rank = wfreq.at(hash).rank;
    occ[rank]++;
    s = fwrite(&rank,sizeof(rank),1,newp);
    if(s!=1) die("Error writing to new parse file");
  }
  if(fclose(newp)!=0) die("Error closing new parse file");
  if(mfclose(moldp)!=0) die("Error closing old parse segment");
  // check old and recomputed occ coincide 
  for(auto& x : wfreq)
    assert(x.second.occ == occ[x.second.rank]);
}
 



void print_help(char** argv, Args &args) {
  cout << "Usage: " << argv[ 0 ] << " <input filename> [options]" << endl;
  cout << "  Options: " << endl
        << "\t-w W\tsliding window size, def. " << args.w << endl
        << "\t-p M\tmodulo for defining phrases, def. " << args.p << endl
        #ifndef NOTHREADS
        << "\t-t M\tnumber of helper threads, def. none " << endl
        #endif        
        << "\t-c  \tdiscard redundant information" << endl
        << "\t-h  \tshow help and exit" << endl
        << "\t-s  \tcompute suffix array info" << endl
        << "\t-r  \tread file in reversed order" << endl;
  #ifdef GZSTREAM
  cout << "If the input file is gzipped it is automatically extracted\n";
  #endif
  exit(1);
}

void parseArgs( int argc, char** argv, Args& arg ) {
   int c;
   extern char *optarg;
   extern int optind;

  puts("==== Command line:");
  for(int i=0;i<argc;i++)
    printf(" %s",argv[i]);
  puts("");

   string sarg;
   while ((c = getopt( argc, argv, "p:w:sht:vcr") ) != -1) {
      switch(c) {
        case 's':
        arg.SAinfo = true; break;
        case 'c':
        arg.compress = true; break;
        case 'r':
        arg.reverse = true; break;
        case 'w':
        sarg.assign( optarg );
        arg.w = stoi( sarg ); break;
        case 'p':
        sarg.assign( optarg );
        arg.p = stoi( sarg ); break;
        case 't':
        sarg.assign( optarg );
        arg.th = stoi( sarg ); break;
        case 'v':
           arg.verbose++; break;
        case 'h':
           print_help(argv, arg); exit(1);
        case '?':
        cout << "Unknown option. Use -h for help." << endl;
        exit(1);
      }
   }
   // the only input parameter is the file name 
   if (argc == optind+1) {
     arg.inputFileName.assign( argv[optind] );
   }
   else {
      cout << "Invalid number of arguments" << endl;
      print_help(argv,arg);
   }
   // check algorithm parameters 
   if(arg.w <4) {
     cout << "Windows size must be at least 4\n";
     exit(1);
   }
   if(arg.p<10) {
     cout << "Modulus must be at leas 10\n";
     exit(1);
   }
   #ifdef NOTHREADS
   if(arg.th!=0) {
     cout << "The NT version cannot use threads\n";
     exit(1);
   }
   #else
   if(arg.th<0) {
     cout << "Number of threads cannot be negative\n";
     exit(1);
   }
   #endif   
}



int main(int argc, char** argv)
{
  // translate command line parameters
  Args arg;
  parseArgs(argc, argv, arg);
  cout << "Windows size: " << arg.w << endl;
  cout << "Stop word modulus: " << arg.p << endl;  

  if (!arg.reverse) cout << "Flag -r is false. Parse in the forward direction." << endl;
  else cout << "Flag -r is true. Parse in the backward direction." << endl;

  // measure elapsed wall clock time
  time_t start_main = time(NULL);
  time_t start_wc = start_main;  
  // init sorted map counting the number of occurrences of each word
  map<uint64_t,word_stats> wordFreq;  
  uint64_t totChar;

  // ------------ parsing input file 
  try {
      if(arg.th==0)
        if (!arg.reverse) totChar = process_file(arg,wordFreq);
        else totChar = process_file_reverse(arg,wordFreq);
      else {
        // NOTE: now newscan.cpp is used only for single threaded parsing, so mt_process_file_reverse is not implemented so far.
        //       pscan.cpp supports multi-threaded parsing for the reversed direction.
        #ifdef NOTHREADS
        cerr << "Sorry, this is the no-threads executable and you requested " << arg.th << " threads\n";
        exit(EXIT_FAILURE);
        #else
        totChar = mt_process_file(arg,wordFreq);
        #endif
      }
  }
  catch(const std::bad_alloc&) {
      cout << "Out of memory (parsing phase)... emergency exit\n";
      die("bad alloc exception");
  }

  // after 1st parsing original text is not required
  // add .rev by default
  if (arg.reverse) arg.inputFileName += ".rev";

  // first report 
  uint64_t totDWord = wordFreq.size();
  cout << "Total input symbols: " << totChar << endl;
  cout << "Found " << totDWord << " distinct words" <<endl;
  cout << "Parsing took: " << difftime(time(NULL),start_wc) << " wall clock seconds\n";  
  // check # distinct words
  if(totDWord>MAX_DISTINCT_WORDS) {
    cerr << "Emergency exit! The number of distinc words (" << totDWord << ")\n";
    cerr << "is larger than the current limit (" << MAX_DISTINCT_WORDS << ")\n";
    exit(1);
  }

  // -------------- second pass  
  start_wc = time(NULL);
  // create array of dictionary words
  vector<const string *> dictArray;
  dictArray.reserve(totDWord);
  // fill array
  uint64_t sumLen = 0;
  uint64_t totWord = 0;
  for (auto& x: wordFreq) {
    sumLen += x.second.str.size();
    totWord += x.second.occ;
    dictArray.push_back(&x.second.str);
  }
  assert(dictArray.size()==totDWord);
  cout << "Sum of lenghts of dictionary words: " << sumLen << endl; 
  cout << "Total number of words: " << totWord << endl; 
  // sort dictionary
  sort(dictArray.begin(), dictArray.end(),pstringCompare);
  // write plain dictionary and occ file, also compute rank for each hash 
  cout << "Writing plain dictionary and occ file\n";
  writeDictOcc(arg, wordFreq, dictArray);
  dictArray.clear(); // reclaim memory
  cout << "Dictionary construction took: " << difftime(time(NULL),start_wc) << " wall clock seconds\n";  
    
  // remap parse file
  start_wc = time(NULL);
  cout << "Generating remapped parse file\n";
  remapParse(arg, wordFreq);
  cout << "Remapping parse file took: " << difftime(time(NULL),start_wc) << " wall clock seconds\n";  
  cout << "==== Elapsed time: " << difftime(time(NULL),start_main) << " wall clock seconds\n";      
  
  return 0;
}

