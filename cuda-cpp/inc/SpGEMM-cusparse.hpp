#pragma once

#include <cusparse_v2.h>

template <class idType, class valType>
cusparseStatus_t SpGEMM_cuSPARSE_numeric(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c);

template <>
cusparseStatus_t SpGEMM_cuSPARSE_numeric<int, float>(CSR<int, float> a, CSR<int, float> b, CSR<int, float> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
{
    return cusparseScsrgemm(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_values, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_values, b.d_rpt, b.d_colids, descr_c, c.d_values, c.d_rpt, c.d_colids);
}

template <>
cusparseStatus_t SpGEMM_cuSPARSE_numeric<int, double>(CSR<int, double> a, CSR<int, double> b, CSR<int, double> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
{
    return cusparseDcsrgemm(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_values, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_values, b.d_rpt, b.d_colids, descr_c, c.d_values, c.d_rpt, c.d_colids);
}

template <class idType, class valType>
void SpGEMM_cuSPARSE_kernel(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c, cusparseHandle_t cusparseHandle, cusparseOperation_t trans_a, cusparseOperation_t trans_b, cusparseMatDescr_t descr_a, cusparseMatDescr_t descr_b, cusparseMatDescr_t descr_c)
{
    cusparseStatus_t status;
    c.nrow = a.nrow;
    c.ncolumn = b.ncolumn;
    c.device_malloc = true;
    cudaMalloc((void **)&(c.d_rpt), sizeof(idType) * (c.nrow + 1));

    status = cusparseXcsrgemmNnz(cusparseHandle, trans_a, trans_b, a.nrow, b.ncolumn, a.ncolumn, descr_a, a.nnz, a.d_rpt, a.d_colids, descr_b, b.nnz, b.d_rpt, b.d_colids, descr_c, c.d_rpt, &(c.nnz));
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "cuSPARSE failed at Symbolic phase" << endl;
    }

    cudaMalloc((void **)&(c.d_colids), sizeof(idType) * (c.nnz));
    cudaMalloc((void **)&(c.d_values), sizeof(valType) * (c.nnz));
        
    status = SpGEMM_cuSPARSE_numeric(a, b, c, cusparseHandle, trans_a, trans_b, descr_a, descr_b, descr_c);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cout << "cuSPARSE failed at Numeric phase" << endl;
    }
}

template <class idType, class valType>
void SpGEMM_cuSPARSE(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{
    cusparseHandle_t cusparseHandle;
    cusparseMatDescr_t descr_a, descr_b, descr_c;
    cusparseOperation_t trans_a, trans_b;

    trans_a = trans_b = CUSPARSE_OPERATION_NON_TRANSPOSE;
  
    /* Set up cuSPARSE Library */
    cusparseCreate(&cusparseHandle);
    cusparseCreateMatDescr(&descr_a);
    cusparseCreateMatDescr(&descr_b);
    cusparseCreateMatDescr(&descr_c);
    cusparseSetMatType(descr_a, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descr_b, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatType(descr_c, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_a, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descr_b, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatIndexBase(descr_c, CUSPARSE_INDEX_BASE_ZERO);
  
    /* Execution of SpMV on Device */
    SpGEMM_cuSPARSE_kernel(a, b, c,
                           cusparseHandle,
                           trans_a, trans_b,
                           descr_a, descr_b, descr_c);
    cudaDeviceSynchronize();
    
    c.memcpyDtH();

    c.release_csr();
    cusparseDestroy(cusparseHandle);
}
