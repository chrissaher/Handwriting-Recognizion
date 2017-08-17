#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int main() {
	string s;
	cin >> s;
	char *cstr = new char[s.length() + 1];
	strcpy(cstr, s.c_str());
	char *pch = strtok(cstr, ",");
	for(int i = 0; i < 28; ++i) {
		for(int j = 0; j < 28; ++j) {
			char c = '.';
			if (atoi(pch) != 255) c = '#';
			cout << c;
			pch = strtok(NULL, ",");
		}
		cout << std::endl;
	}
	delete [] cstr;
	return 0;
}