#include <vector>
#include <cmath>
#include <algorithm>

class Node {
public:
    Node(Node* parent=nullptr, int action=0, int state=0) :
        parent(parent), action(action), state(state), visits(0), rewards(0) {}

    Node* add_child(int action, int state) {
        Node* child = new Node(this, action, state);
        children.push_back(child);
        return child;
    }

    Node* select_child() {
        // Select the child with the highest UCB score
        std::vector<double> ucb_scores;
        for (Node* child : children) {
            ucb_scores.push_back(
                (child->rewards / child->visits) +
                std::sqrt((2 * std::log(visits)) / child->visits)
            );
        }
        auto result = std::max_element(ucb_scores.begin(), ucb_scores.end());
        return children[std::distance(ucb_scores.begin(), result)];
    }

    void update(double rewards) {
        visits += 1;
        this->rewards += rewards;
    }

private:
    Node* parent;
    int action;
    int state;
    std::vector<Node*> children;
    int visits;
    double rewards;
};