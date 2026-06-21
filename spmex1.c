#include <math.h>
#include <string.h>
#include "matrix.h"
#include "mex.h"

// REFERENCES
// https://www.mathworks.com/help/matlab/matlab_external/gateway-routine.html
// https://www.mathworks.com/help/matlab/cc-mx-matrix-library.html

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxSetNzmin(mxArray* spA, mwSize nzmin) {
	mwSize nzmax = mxGetNzmax(spA);

	if (nzmin > nzmax) {
		if (nzmin < nzmax << 1) {
			nzmin = nzmax << 1;
		}

		mxSetNzmax(spA, nzmin);
		mwIndex* ir = mxGetIr(spA);
		ir = mxRealloc(ir, nzmin * sizeof(mwIndex));
		mxSetIr(spA, ir);

		mxDouble* nz = mxGetDoubles(spA);
		nz = mxRealloc(nz, nzmin * sizeof(mxDouble));
		mxSetDoubles(spA, nz);
	}
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxMoveVal(mxArray* vec, mwIndex pos1, mwIndex pos2) {
	if (pos1 > pos2) {
		mxDouble* real = mxGetDoubles(vec);
		mxDouble temp = real[pos1];
		size_t bytes = (pos1 - pos2) * sizeof(mxDouble);
		memmove(&real[pos2 + 1], &real[pos2], bytes);
		real[pos2] = temp;
	}
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxMoveCol(mxArray* spA, mwIndex col1, mwIndex col2) {
	if (col1 > col2) {
		mwIndex* jc = mxGetJc(spA);
		mwSize nnz = jc[mxGetN(spA)];
		mwSize nnz1 = jc[col1 + 1] - jc[col1];
		mxSetNzmin(spA, nnz + nnz1);

		mwIndex ka = jc[col2];
		mwIndex kb = ka + nnz1;
		mwIndex kc = jc[col1];
		mwSize nnz2 = kc - ka;

		mwIndex* ir = mxGetIr(spA);
		memcpy(&ir[nnz], &ir[kc], nnz1 * sizeof(mwIndex));
		memmove(&ir[kb], &ir[ka], nnz2 * sizeof(mwIndex));
		memcpy(&ir[ka], &ir[nnz], nnz1 * sizeof(mwIndex));

		mxDouble* nz = mxGetDoubles(spA);
		memcpy(&nz[nnz], &nz[kc], nnz1 * sizeof(mxDouble));
		memmove(&nz[kb], &nz[ka], nnz2 * sizeof(mxDouble));
		memcpy(&nz[ka], &nz[nnz], nnz1 * sizeof(mxDouble));

		for (mwIndex j = col1; j > col2; j--) {
			jc[j] = jc[j - 1] + nnz1;
		}
	}
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxMoveRow(mxArray* spA, mwIndex row1, mwIndex row2) {
	if (row1 > row2) {
		mwIndex* jc = mxGetJc(spA);
		mwIndex ncol = mxGetN(spA);
		mwIndex* ir = mxGetIr(spA);

		for (mwIndex j = 0; j < ncol; j++) {
			mwIndex k2 = jc[j];

			while (k2 < jc[j + 1]) {
				if (ir[k2] >= row2) {
					break; // while
				}
				k2++;
			}

			mwIndex k1 = k2;
			bool isgt = false;

			while (k1 < jc[j + 1]) {
				if (ir[k1] >= row1) {
					if (ir[k1] == row1) {
						ir[k1] = row2;
						isgt = k1 > k2;
					}
					break; // while
				}
				ir[k1]++;
				k1++;
			}

			if (isgt) {
				mwSize nnz = k1 - k2;
				mwIndex k3 = k2 + 1;
				mxAssert(row2 == ir[k1], "");
				memmove(&ir[k3], &ir[k2], nnz * sizeof(mwIndex));
				ir[k2] = row2;

				mxDouble* nz = mxGetDoubles(spA);
				mxDouble temp = nz[k1];
				memmove(&nz[k3], &nz[k2], nnz * sizeof(mxDouble));
				nz[k2] = temp;
			}
		}
	}
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxBSXTimes(mxArray* spC, const mxArray* spA, const mxArray* spB,
	char flag) {
	mwIndex mrowA = mxGetM(spA);
	mwIndex ncolB = mxGetN(spB);

	mwIndex* jcC = mxGetJc(spC);
	mwIndex* jcA = mxGetJc(spA);
	mwIndex* jcB = mxGetJc(spB);

	mwIndex* irA = mxGetIr(spA);
	mwIndex* irB = mxGetIr(spB);

	for (mwIndex j = 0; j < ncolB; j++) {
		mwSize nnza = jcA[j + 1] - jcA[j];
		mwSize nnzb = jcB[j + 1] - jcB[j];
		mwSize nnzc = nnza * nnzb;

		mwIndex kC = jcC[j];
		mxSetNzmin(spC, kC + nnzc);
		mwIndex* irC = mxGetIr(spC);

		mxDouble* nzC = mxGetDoubles(spC);
		mxDouble* nzA = mxGetDoubles(spA);
		mxDouble* nzB = mxGetDoubles(spB);

		for (mwIndex kB = jcB[j]; kB < jcB[j + 1]; kB++) {
			mwIndex rowC = irB[kB] * mrowA;

			for (mwIndex kA = jcA[j]; kA < jcA[j + 1]; kA++) {
				irC[kC] = irA[kA] + rowC;

				switch (flag) {
				case '*':
					nzC[kC] = nzA[kA] * nzB[kB];
					break;
				case '/':
					nzC[kC] = nzA[kA] / nzB[kB];
					break;
				case '\\':
					nzC[kC] = nzB[kB] / nzA[kA];
					break;
				}

				if (nzC[kC] != 0.0) kC++;
			}
		}

		jcC[j + 1] = kC;
	}
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxBSXPlus(mxArray* spC, const mxArray* spA, const mxArray* spB,
	char flag) {
	mwIndex mrowC = mxGetM(spC);
	mwIndex mrowA = mxGetM(spA);
	mwIndex mrowB = mxGetM(spB);
	mwIndex ncol = mxGetN(spC);

	mwIndex* jcC = mxGetJc(spC);
	mwIndex* jcA = mxGetJc(spA);
	mwIndex* jcB = mxGetJc(spB);

	mwIndex* irA = mxGetIr(spA);
	mwIndex* irB = mxGetIr(spB);

	for (mwIndex j = 0; j < ncol; j++) {
		mwSize nnza = jcA[j + 1] - jcA[j];
		mwSize nnzb = jcB[j + 1] - jcB[j];
		mwSize noza = mrowA - nnza;
		mwSize nozb = mrowB - nnzb;

		mwSize nnzc = mrowC - noza * nozb;
		mwIndex kC = jcC[j];
		mxSetNzmin(spC, kC + nnzc);
		mwIndex* irC = mxGetIr(spC);

		mxDouble* nzC = mxGetDoubles(spC);
		mxDouble* nzA = mxGetDoubles(spA);
		mxDouble* nzB = mxGetDoubles(spB);

		mwIndex rowB = 0;
		mwIndex kB = jcB[j];

		while (rowB < mrowB) {
			mwIndex rowC = mrowA * rowB;

			if (kB < jcB[j + 1] && rowB == irB[kB]) {
				mwIndex kA = jcA[j];

				for (mwIndex rowA = 0; rowA < mrowA; rowA++) {
					irC[kC] = rowA + rowC;

					switch (flag) {
					case '+':
						if (kA < jcA[j + 1] && rowA == irA[kA]) {
							nzC[kC] = nzA[kA] + nzB[kB];
							if (nzC[kC] != 0.0) kC++;
							kA++;
						}
						else {
							nzC[kC++] = +nzB[kB];
						}
						break;
					case '-':
						if (kA < jcA[j + 1] && rowA == irA[kA]) {
							nzC[kC] = nzA[kA] - nzB[kB];
							if (nzC[kC] != 0.0) kC++;
							kA++;
						}
						else {
							nzC[kC++] = -nzB[kB];
						}
						break;
					}
				}

				rowB++;
				kB++;
			}
			else if (nnza > 0) {
				for (mwIndex kA = jcA[j]; kA < jcA[j + 1]; kA++) {
					irC[kC] = irA[kA] + rowC;
					nzC[kC++] = nzA[kA];
				}

				rowB++;
			}
			else {
				rowB = kB < jcB[j + 1] ? irB[kB] : mrowB;
			}
		}

		jcC[j + 1] = kC;
	}
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static mxArray* mxPivotSeq(mwIndex ncol) {
	mxArray* seq = mxCreateDoubleMatrix(1, ncol, mxREAL);
	mxAssert(seq != NULL, "");
	mxDouble* real = mxGetDoubles(seq);

	for (mwIndex col = 0; col < ncol; col++) {
		real[col] = (mxDouble)(col + 1);
	}

	return seq;
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxFinishL(mxArray* spL, mwIndex pos) {
	mwIndex ncol = mxGetN(spL);
	mwIndex mrow = mxGetM(spL);

	if (pos < ncol && pos < mrow) {
		mwIndex* jc = mxGetJc(spL);
		mxAssert(jc[pos] == jc[ncol], "");

		for (mwIndex col = pos; col < ncol; col++) {
			jc[col + 1] = jc[col] + 1;
		}

		mwSize nnz = jc[ncol];
		mxSetNzmin(spL, nnz);
		mwIndex* ir = mxGetIr(spL);
		mwIndex row = pos;

		mxDouble* nz = mxGetDoubles(spL);

		for (mwIndex k = jc[pos]; k < nnz; k++) {
			ir[k] = row++;
			nz[k] = (mxDouble){ 1 };
		}
	}
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxUseDeltaA(mxArray* spU, const mxArray* spA, const mxArray* spDA,
	mwIndex row, mwIndex col) {
	mwIndex mrow = mxGetM(spA);
	mwIndex ncol = mxGetN(spA);

	mxAssert(mrow == mxGetM(spDA) && ncol == mxGetN(spDA), "");
	mxAssert(mrow == mxGetM(spU) && ncol == mxGetN(spU), "");
	mxAssert(row < mrow && col < ncol, "");

	mwIndex* jcU = mxGetJc(spU);
	mwIndex* jcA = mxGetJc(spA);
	memcpy(jcU, jcA, (col + 1) * sizeof(mwIndex));
	mwIndex* irA = mxGetIr(spA);
	mwIndex kU;

	for (kU = jcA[col]; kU < jcA[col + 1]; kU++) {
		if (irA[kU] > row) {
			break; // for
		}
	}

	jcU[col + 1] = kU;
	mwIndex* jcDA = mxGetJc(spDA);

	for (mwIndex j = col + 1; j < ncol; j++) {
		mwSize nnzA = jcA[j + 1] - jcA[j];
		mwSize nnzDA = jcDA[j + 1] - jcDA[j];
		mwSize nnzU = nnzA + nnzDA;
		if (nnzU > mrow) nnzU = mrow;
		jcU[j + 1] = jcU[j] + nnzU;
	}

	mxSetNzmin(spU, jcU[ncol]);
	mwIndex* irU = mxGetIr(spU);
	memcpy(irU, irA, kU * sizeof(mwIndex));
	mwIndex* irDA = mxGetIr(spDA);

	mxDouble* nzU = mxGetDoubles(spU);
	mxDouble* nzA = mxGetDoubles(spA);
	memcpy(nzU, nzA, kU * sizeof(mxDouble));
	mxDouble* nzDA = mxGetDoubles(spDA);

	for (mwIndex j = col + 1; j < ncol; j++) {
		mwIndex kA = jcA[j];
		mwSize nnzA = jcA[j + 1] - kA;
		mwIndex kDA = jcDA[j];
		mwSize nnzDA = jcDA[j + 1] - kDA;

		if (nnzA > 0 && nnzDA > 0) {
			while (kA < jcA[j + 1] && kDA < jcDA[j + 1]) {
				if (irA[kA] < irDA[kDA]) {
					irU[kU] = irA[kA];
					nzU[kU] = nzA[kA];
					kU++;
					kA++;
				}
				else if (irDA[kDA] < irA[kA]) {
					irU[kU] = irDA[kDA];
					nzU[kU] = nzDA[kDA];
					kU++;
					kDA++;
				}
				else {
					mxDouble sum = nzA[kA] + nzDA[kDA];
					if (sum != 0.0) {
						irU[kU] = irA[kA];
						nzU[kU] = sum;
						kU++;
					}
					kA++;
					kDA++;
				}
			}

			nnzA = jcA[j + 1] - kA;
			nnzDA = jcDA[j + 1] - kDA;
		}

		if (nnzA > 0) {
			mxAssert(nnzDA == 0, "");
			memcpy(&irU[kU], &irA[kA], nnzA * sizeof(mwIndex));
			memcpy(&nzU[kU], &nzA[kA], nnzA * sizeof(mxDouble));
			kU += nnzA;
		}
		else if (nnzDA > 0) {
			mxAssert(nnzA == 0, "");
			memcpy(&irU[kU], &irDA[kDA], nnzDA * sizeof(mwIndex));
			memcpy(&nzU[kU], &nzDA[kDA], nnzDA * sizeof(mxDouble));
			kU += nnzDA;
		}

		jcU[j + 1] = kU;
	}
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxSetDeltaA(mxArray* spDA, const mxArray* spA,
	mwIndex row, mwIndex col) {
	mwIndex mrow = mxGetM(spA);
	mwIndex ncol = mxGetN(spA);

	mxAssert(mrow == mxGetM(spDA) && ncol == mxGetN(spDA), "");
	mxAssert(row < mrow && col < ncol, "");

	mwIndex* jcA = mxGetJc(spA);
	mwIndex* irA = mxGetIr(spA);
	mwIndex kdivA;

	for (kdivA = jcA[col]; kdivA < jcA[col + 1]; kdivA++) {
		if (irA[kdivA] == row) {
			break; // for
		}
	}

	mxAssert(kdivA < jcA[col + 1], "");
	mwIndex nnzr = jcA[col + 1] - kdivA - 1;
	mwIndex* jcDA = mxGetJc(spDA);

	if (nnzr > 0) {
		memset(jcDA, 0, (col + 1) * sizeof(mwIndex));
		jcDA[col + 1] = nnzr;

		for (mwIndex j = col + 1; j < ncol; j++) {
			jcDA[j + 1] = jcDA[j];

			for (mwIndex kA = jcA[j]; kA < jcA[j + 1]; kA++) {
				if (irA[kA] >= row) {
					if (irA[kA] == row) {
						jcDA[j + 1] += nnzr;
					}
					break; // for
				}
			}
		}

		mxSetNzmin(spDA, jcDA[ncol]);
		mwIndex* irDA = mxGetIr(spDA);
		mwIndex kA = kdivA + 1;
		mwIndex kDA;

		for (mwIndex j = col; j < ncol; j++) {
			kDA = jcDA[j];

			if (jcDA[j + 1] > kDA) {
				memcpy(&irDA[kDA], &irA[kA], nnzr * sizeof(mwIndex));
			}
		}

		mxDouble* nzDA = mxGetDoubles(spDA);
		mxDouble* nzA = mxGetDoubles(spA);

		for (kDA = 0; kDA < nnzr; kDA++) {
			nzDA[kDA] = nzA[kA] / nzA[kdivA];
			kA++;
		}

		for (mwIndex j = col + 1; j < ncol; j++) {
			if (jcDA[j + 1] > kDA) {
				for (kA = jcA[j]; kA < jcA[j + 1]; kA++) {
					if (irA[kA] == row) {
						break; // for
					}
				}

				mxAssert(kA < jcA[j + 1], "");

				for (mwIndex kDA_ = 0; kDA_ < nnzr; kDA_++) {
					nzDA[kDA] = -nzDA[kDA_] * nzA[kA];
					kDA++;
				}
			}
		}
	}
	else {
		memset(jcDA, 0, (ncol + 1) * sizeof(mwIndex));
	}
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxSetLCol(mxArray* spL, const mxArray* spDA, mwIndex pos) {
	mwIndex mrow = mxGetM(spL);
	mwIndex ncol = mxGetN(spDA);
	mxAssert(mxGetN(spL) == mrow && mxGetM(spDA) == mrow, "");
	mxAssert(pos < mrow && pos < ncol, "");

	mwIndex* jcL = mxGetJc(spL);
	mwIndex kL = jcL[pos];
	mxAssert(kL == jcL[mrow], "");
	mwIndex* jcDU = mxGetJc(spDA);
	mxAssert(jcDU[pos] == 0, "");
	mwSize nnz = jcDU[pos + 1];
	jcL[pos + 1] = kL + 1 + nnz;

	for (mwIndex col = pos + 1; col < mrow; col++) {
		jcL[col + 1] = jcL[col];
	}

	mxSetNzmin(spL, jcL[mrow]);
	mwIndex* irL = mxGetIr(spL);
	irL[kL] = pos;
	mwIndex* irDA = mxGetIr(spDA);
	memcpy(&irL[kL + 1], &irDA[0], nnz * sizeof(mwIndex));

	mxDouble* nzL = mxGetDoubles(spL);
	mxDouble* nzDA = mxGetDoubles(spDA);
	nzL[kL] = (mxDouble){ 1 };
	memcpy(&nzL[kL + 1], &nzDA[0], nnz * sizeof(mxDouble));
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static bool mxGetPivot(const mxArray* spA, mwIndex row, mwIndex col,
	mwIndex* ipiv, mwIndex* jpiv) {
	mwIndex mrow = mxGetM(spA);
	mwIndex ncol = mxGetN(spA);
	mxAssert(row < mrow && col < ncol, "");

	mwIndex* jc = mxGetJc(spA);
	mwIndex* ir = mxGetIr(spA);
	mxDouble maxval = 0.0;

	for (mwIndex j = col; j < ncol; j++) {
		mwIndex kpiv;

		for (kpiv = jc[j]; kpiv < jc[j + 1]; kpiv++) {
			if (ir[kpiv] >= row) {
				break; // for
			}
		}

		if (kpiv < jc[j + 1]) {
			mxDouble* nz = mxGetDoubles(spA);
			mxDouble val = fabs(nz[kpiv]);

			for (mwIndex k = kpiv + 1; k < jc[j + 1]; k++) {
				mxDouble chk = fabs(nz[k]);

				if (chk > val) {
					val = chk;
					kpiv = k;
				}
			}

			if (val > maxval) {
				maxval = val;
				*ipiv = ir[kpiv];
				*jpiv = j;
			}
		}
	}

	return maxval > 0.0;
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxCreateBSX(mxArray* plhs[], const mxArray* prhs[], char flag) {
	const mxArray* spA = prhs[2];
	const mxArray* spB = prhs[3];
	mwIndex mrow = mxGetM(spA) * mxGetM(spB);
	mwIndex ncol = mxGetN(spA);

	mwSize nzmax = mxGetJc(spA)[ncol] + mxGetJc(spB)[ncol];
	nzmax = nzmax <= mrow * ncol ? nzmax : mrow * ncol;
	nzmax = nzmax > 0 ? nzmax : 1;
	mxArray* spC = mxCreateSparse(mrow, ncol, nzmax, mxREAL);
	mxAssert(spC != NULL, "");
	memset(mxGetJc(spC), 0, (ncol + 1) * sizeof(mwIndex));

	switch (flag) {
	case '+':
	case '-':
		mxBSXPlus(spC, spA, spB, flag);
		break;
	default:
		mxBSXTimes(spC, spA, spB, flag);
		break;
	}

	plhs[0] = spC;
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static char mxCheckBSX(int nlhs, int nrhs, const mxArray* prhs[]) {
	if (nrhs < 2) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"A second input argument is required.");
	}
	if (!mxIsChar(prhs[1])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"Argument two must be a character array.");
	}

	if (nrhs < 3) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"A third input argument is required.");
	}
	if (!mxIsNumeric(prhs[2])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"Argument three must be a numeric array.");
	}
	if (!mxIsSparse(prhs[2])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"Argument three must be a sparse array.");
	}
	if (mxIsComplex(prhs[2])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"Argument three must be a real array.");
	}

	if (nrhs < 4) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"A fourth input argument is required.");
	}
	if (!mxIsNumeric(prhs[3])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"Argument four must be a numeric array.");
	}
	if (!mxIsSparse(prhs[3])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"Argument four must be a sparse array.");
	}
	if (mxIsComplex(prhs[3])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"Argument four must be a real array.");
	}

	char fun[sizeof("rdivide")];
	char flag = '?';

	if (mxGetString(prhs[1], fun, sizeof(fun)) == 0) {
		if (strcmp(fun, "plus") == 0 || strcmp(fun, "+") == 0) {
			flag = '+';
		}
		else if (strcmp(fun, "minus") == 0 || strcmp(fun, "-") == 0) {
			flag = '-';
		}
		else if (strcmp(fun, "times") == 0 || strcmp(fun, ".*") == 0) {
			flag = '*';
		}
		else if (strcmp(fun, "rdivide") == 0 || strcmp(fun, "./") == 0) {
			flag = '/';
		}
		else if (strcmp(fun, "ldivide") == 0 || strcmp(fun, ".\\") == 0) {
			flag = '\\';
		}
	}

	if (mxGetN(prhs[2]) != mxGetN(prhs[3])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"Sparse operands must have equal column sizes.");
	}

	switch (flag) {
	case '/':
		if (mxGetJc(prhs[3])[mxGetN(prhs[3])] < mxGetNumberOfElements(prhs[3])) {
			mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
				"Ensure right-division divisor is all nonzero.");
		}
		break;
	case '\\':
		if (mxGetJc(prhs[2])[mxGetN(prhs[2])] < mxGetNumberOfElements(prhs[2])) {
			mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
				"Ensure left-division divisor is all nonzero.");
		}
		break;
	case '?':
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"Use 'plus', 'minus', 'times', 'rdivide', 'ldivide',\n"
			"or equivalent operator symbol(s) as argument two.");
		break;
	}

	if (nrhs > 4) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"Only four input arguments are allowed.");
	}
	if (nlhs > 1) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckBSX",
			"One output argument is allowed.");
	}

	return flag;
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxCreateRRLU(int nlhs, mxArray* plhs[], const mxArray* prhs[]) {
	mxArray* spU1 = mxDuplicateArray(prhs[1]);
	mxAssert(spU1 != NULL, "");
	bool isU1 = true;

	mwIndex mrow = mxGetM(spU1);
	mwIndex ncol = mxGetN(spU1);

	mxArray* spU2 = mxCreateSparse(mrow, ncol, 1, mxREAL);
	mxAssert(spU2 != NULL, "");
	memset(mxGetJc(spU2), 0, (ncol + 1) * sizeof(mwIndex));

	mxArray* spDA = mxCreateSparse(mrow, ncol, 1, mxREAL);
	mxAssert(spDA != NULL, "");
	memset(mxGetJc(spDA), 0, (ncol + 1) * sizeof(mwIndex));

	plhs[0] = mxCreateSparse(mrow, mrow, 1, mxREAL);
	mxAssert(plhs[0] != NULL, "");
	memset(mxGetJc(plhs[0]), 0, (mrow + 1) * sizeof(mwIndex));
	mwIndex k;

	if (nlhs > 2) {
		plhs[2] = mxPivotSeq(mrow);

		if (nlhs > 3) {
			plhs[3] = mxPivotSeq(ncol);
		}
	}

	for (k = 0; k < mrow && k < ncol; k++) {
		mwIndex i, j;

		if (isU1) {
			if (mxGetPivot(spU1, k, k, &i, &j)) {
				mxMoveRow(plhs[0], i, k);
				mxMoveRow(spU1, i, k);
				mxMoveCol(spU1, j, k);
				mxSetDeltaA(spDA, spU1, k, k);
				mxSetLCol(plhs[0], spDA, k);
				mxUseDeltaA(spU2, spU1, spDA, k, k);
				isU1 = false;
			}
			else {
				break; // for
			}
		}
		else {
			if (mxGetPivot(spU2, k, k, &i, &j)) {
				mxMoveRow(plhs[0], i, k);
				mxMoveRow(spU2, i, k);
				mxMoveCol(spU2, j, k);
				mxSetDeltaA(spDA, spU2, k, k);
				mxSetLCol(plhs[0], spDA, k);
				mxUseDeltaA(spU1, spU2, spDA, k, k);
				isU1 = true;
			}
			else {
				break; // for
			}
		}

		if (nlhs > 2) {
			mxMoveVal(plhs[2], i, k);

			if (nlhs > 3) {
				mxMoveVal(plhs[3], j, k);
			}
		}
	}

	mxFinishL(plhs[0], k);

	if (nlhs > 1) {
		if (isU1) {
			plhs[1] = spU1;
			mxDestroyArray(spU2);
		}
		else {
			plhs[1] = spU2;
			mxDestroyArray(spU1);
		}
	}
	else {
		mxDestroyArray(spU1);
		mxDestroyArray(spU2);
	}

	mxDestroyArray(spDA);
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxCheckRRLU(int nlhs, int nrhs, const mxArray* prhs[]) {
	if (nrhs < 2) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckRRLU",
			"A second input argument is required.");
	}
	if (!mxIsNumeric(prhs[1])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckRRLU",
			"Argument two must be a numeric array.");
	}
	if (!mxIsSparse(prhs[1])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckRRLU",
			"Argument two must be a sparse array.");
	}
	if (mxIsComplex(prhs[1])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckRRLU",
			"Argument two must be a real array.");
	}

	if (nrhs > 2) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckRRLU",
			"Only two input arguments are allowed.");
	}
	if (nlhs > 4) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckRRLU",
			"Four output arguments are allowed.");
	}
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxPrintDebug(const mxArray* prhs[]) {
	const mxArray* spA = prhs[1];
	mwIndex* ir = mxGetIr(spA);
	mwIndex* jc = mxGetJc(spA);
	mwIndex ncol = mxGetN(spA);
	mwIndex nnz = jc[ncol];

	if (mxIsComplex(spA)) {
		mexPrintf("%8s%8s%8s%16s%16s\n", "k", "ir", "jc",
			"nz.real", "nz.imag");
		mxComplexDouble* nz = mxGetComplexDoubles(spA);

		for (mwIndex k = 0; k < nnz; k++) {
			if (k <= ncol) {
				mexPrintf("%8d%8d%8d%16g%16g\n", k, ir[k], jc[k],
					nz[k].real, nz[k].imag);
			}
			else {
				mexPrintf("%8d%8d%8c%16g%16g\n", k, ir[k], '-',
					nz[k].real, nz[k].imag);
			}
		}

		for (mwIndex k = nnz; k <= ncol; k++) {
			mexPrintf("%8c%8c%8d%16c%16c\n", '-', '-', jc[k], '-', '-');
		}
	}
	else {
		mexPrintf("%8s%8s%8s%16s\n", "k", "ir", "jc", "nz");
		mxDouble* nz = mxGetDoubles(spA);

		for (mwIndex k = 0; k < nnz; k++) {
			if (k <= ncol) {
				mexPrintf("%8d%8d%8d%16g\n", k, ir[k], jc[k], nz[k]);
			}
			else {
				mexPrintf("%8d%8d%8c%16g\n", k, ir[k], '-', nz[k]);
			}
		}

		for (mwIndex k = nnz; k <= ncol; k++) {
			mexPrintf("%8c%8c%8d%16c\n", '-', '-', jc[k], '-');
		}
	}
}
#endif

#if MX_HAS_INTERLEAVED_COMPLEX
static void mxCheckDebug(int nlhs, int nrhs, const mxArray* prhs[]) {
	if (nrhs < 2) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckDebug",
			"A second input argument is required.");
	}
	if (!mxIsNumeric(prhs[1])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckDebug",
			"Argument two must be a numeric array.");
	}
	if (!mxIsSparse(prhs[1])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckDebug",
			"Argument two must be a sparse array.");
	}

	if (nrhs > 2) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckDebug",
			"Only two input arguments are allowed.");
	}
	if (nlhs > 0) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mxCheckDebug",
			"No output arguments are allowed.");
	}
}
#endif

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
#if MX_HAS_INTERLEAVED_COMPLEX
	if (nrhs < 1) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mexFunction",
			"At least one input argument is required.");
	}
	if (!mxIsChar(prhs[0])) {
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mexFunction",
			"Argument one must be a character array.");
	}

	char* arg = mxArrayToString(prhs[0]);
	mxAssert(arg != NULL, "");

	if (strcmp(arg, "debug") == 0) {
		mxFree(arg);
		mxCheckDebug(nlhs, nrhs, prhs);
		mxPrintDebug(prhs);
	}
	else if (strcmp(arg, "rrlu") == 0) {
		mxFree(arg);
#ifndef rrlu
		mxCheckRRLU(nlhs, nrhs, prhs);
		mxCreateRRLU(nlhs, plhs, prhs);
#else
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mexFunction",
			"The 'rrlu' use case is disabled via -Drrlu.");
#endif
	}
	else if (strcmp(arg, "bsx") == 0) {
		mxFree(arg);
#ifndef bsx
		char flag = mxCheckBSX(nlhs, nrhs, prhs);
		mxCreateBSX(plhs, prhs, flag);
#else
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mexFunction",
			"The 'bsx' use case is disabled via -Dbsx.");
#endif
	}
	else {
		mxFree(arg);
		mexErrMsgIdAndTxt("RTToolbox:spmex1:mexFunction",
			"Use 'debug', 'rrlu', or 'bsx' as argument one.");
	}
#else
	mexErrMsgIdAndTxt("RTToolbox:spmex1:mexFunction",
		"Build for MX_HAS_INTERLEAVED_COMPLEX.");
#endif
}
