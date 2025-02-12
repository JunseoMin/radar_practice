#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <pmc/pmc.h>

using namespace std;
using namespace Eigen;

// Eigen::MatrixXd를 CSR 형식으로 변환하는 함수
void convertToCSR(const MatrixXd& G, vector<double>& values, vector<int>& column_indices, vector<int>& row_ptr) {
    int rows = G.rows();
    int cols = G.cols();
    
    row_ptr.push_back(0); // 첫 번째 행의 시작 인덱스

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (G(i, j) != 0) { // 0이 아닌 요소만 저장
                values.push_back(G(i, j));
                column_indices.push_back(j);
            }
        }
        row_ptr.push_back(values.size()); // 다음 행의 시작 인덱스
    }
}

pmc::pmc_graph EigenToPMC(const Eigen::MatrixXd& G){
  vector<long long> vs;
  vector<int>       es;

  vs.push_back(0);

  int tmp = 0;

  for (int vertex = 0; vertex < G.rows(); vertex++)
  {
    int n_edges =  G.row(vertex).sum();
    tmp += n_edges;
    vs.push_back(tmp);
    for (int edge_idx = 0; edge_idx < G.cols(); edge_idx ++){
      if (G(vertex,edge_idx) == 1)
      {
        es.push_back(edge_idx);
      }
    }
  }
  
  // std::cout << "vs: " << std::endl;
  // for (int v : vs) cout << v << " ";
  // std::cout << std::endl;
  // std::cout << "es: " << std::endl;
  // for (int e : es) cout << e << " ";
  // std::cout << std::endl;

  pmc::pmc_graph p_G(vs,es);
  return p_G;
}
 
int main() {
    // 예제 행렬 (희소 행렬)
    MatrixXd G(4, 4);
    G << 0, 1, 0, 1,
         1, 0, 1, 1,
         0, 1, 0, 0,
         1, 1, 0, 0;

    vector<double> values;
    vector<int> column_indices;
    vector<int> row_ptr;

    std::cout << "original edges: " << G.sum() / 2 << std::endl;
    std::cout << "original verti: " << G.rows() << std::endl;
    convertToCSR(G, values, column_indices, row_ptr);
    pmc::pmc_graph pg = EigenToPMC(G);

    std::cout << "calc num edges: " << pg.num_edges() << std::endl;
    std::cout << "calc num verti: " << pg.num_vertices() << std::endl;

    return 0;
}
