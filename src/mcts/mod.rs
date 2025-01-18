// Monte-Carlo Tree Search-driven reinforcement learning as implemented in
// AlphaZero and described here: https://github.com/asdfjkl/neural_network_chess/releases/download/v1.6/neural_networks_chess.pdf

use crate::{board::{Board, WHITE}, chess_move::ChessMove};
use pyo3::prelude::*;

#[derive(Clone)]
struct Mcts {
    pub nodes: Vec<MctsNode>,
    pub edges: Vec<MctsEdge>,
    pub c: f32,
    pub to_move: u64,
}

#[derive(Clone, Copy)]
struct MctsEdge {
    pub parent_node_idx: usize,
    pub chess_move: ChessMove,
    pub n: usize, 
    pub w: f32,
    pub q: f32, 
    pub p: f32,
}

#[derive(Clone)]
struct MctsNode {
    pub self_node_idx: usize,
    pub parent_edge_idx: usize,
    pub children: Vec<(usize, Option<usize>)>,
}

impl Mcts {
    pub fn new(c: f32, to_move: u64) -> Self {
        Self {
            nodes: vec![],
            edges: vec![],
            c,
            to_move,
        }
    }

    pub fn uct_value(&self, edge_idx: usize, parent_n: usize) -> f32 {
        let edge = self.edges[edge_idx];
        self.c * edge.p * (parent_n as f32).sqrt() / (1f32 + edge.n as f32)
    }

    pub fn back_propagate(&mut self, eval: f32, parent_edge_idx: usize, to_move: u64) {
        if parent_edge_idx == usize::MAX {
            return;
        }

        self.edges[parent_edge_idx].n += 1;
        self.edges[parent_edge_idx].w += eval;
        self.edges[parent_edge_idx].q = self.edges[parent_edge_idx].w / self.edges[parent_edge_idx].n as f32;

        let color = if self.to_move == to_move { -1f32 } else { 1f32 };

        if self.edges[parent_edge_idx].parent_node_idx != usize::MAX {
            let parent_idx = self.edges[parent_edge_idx].parent_node_idx;
            if self.nodes[parent_idx].parent_edge_idx != usize::MAX {
                self.back_propagate(eval * color, self.nodes[parent_idx].parent_edge_idx, to_move);
            }
        }
    }
}

impl MctsEdge {
    pub fn new(parent_node_idx: usize, chess_move: ChessMove) -> Self {
        Self {
            parent_node_idx: parent_node_idx,
            chess_move: chess_move,
            n: 0,
            w: 0f32,
            q: 0f32,
            p: 0f32,
        }
    }
}

impl MctsNode {
    pub fn new(board: &mut Board, parent_edge_idx: usize, mcts: &mut Mcts) -> usize {
        let self_node_idx = mcts.nodes.len();
        
        let color_global = if mcts.to_move == board.to_move() { 1f32 } else { -1f32 };
        let mut eval = board.forward().unwrap() * color_global;
        
        let mut children = vec![];

        let moves = board.moves();
        if moves.is_empty() {
            if board.check() {
                eval = -1000f32 * color_global;
            } else {
                eval = 0f32;
            }
        } else {
            children = board.moves()
                .into_iter()
                .map(|m| {
                    let edge_idx = mcts.edges.len();
                    mcts.edges.push(MctsEdge::new(self_node_idx, m));
                    edge_idx
                })
                .map(|e| (e, None))
                .collect::<Vec<_>>();

            let color = if board.to_move() == WHITE { -1f32 } else { 1f32 };
            let child_evals = children.iter()
                .map(|(e, _)| mcts.edges[*e])
                .map(|m| {
                    board.push_move(m.chess_move);
                    let eval_child = board.forward().unwrap() * color;
                    board.pop_move();
                    eval_child.exp()
                })
                .collect::<Vec<_>>();

            let sum: f32 = (child_evals.iter().sum::<f32>() + 0.000001).min(f32::MAX);
            for (i, (edge, _)) in children.iter().enumerate() {
                mcts.edges[*edge].p = child_evals[i] / sum;
            }
        }

        let node = Self {
            self_node_idx: self_node_idx,
            parent_edge_idx: parent_edge_idx,
            children: children,
        };

        mcts.nodes.push(node);

        mcts.back_propagate(eval, parent_edge_idx, board.to_move());

        self_node_idx
    }

    pub fn select(&self, board: &mut Board, mcts: &mut Mcts) {
        let (mut max_uct_child, mut max_uct_value, mut max_uct_edge, mut child_idx) 
            = (usize::MAX, f32::MIN, None, usize::MAX);
        let color = if mcts.to_move == board.to_move() { 1f32 }  else { -1f32 };

        if self.children.is_empty() {
            let mut eval = 0f32;
            if board.check() {
                eval = -1000f32 * color;
            }
            mcts.back_propagate(eval, self.parent_edge_idx, board.to_move());
            return;
        }

        for (i, (edge, child)) in self.children.iter().enumerate() {
            let parent_edge_n = if self.parent_edge_idx == usize::MAX { 1 } else { mcts.edges[self.parent_edge_idx].n };
            let q = mcts.edges[*edge].q;

            let uct_value = mcts.uct_value(*edge, parent_edge_n) + q * color;
            if uct_value > max_uct_value {
                max_uct_value = uct_value;
                max_uct_edge = Some(edge);
                child_idx = i;
                if let Some(child_node) = child { 
                    max_uct_child = *child_node 
                } else {
                    max_uct_child = usize::MAX;
                };
            }
        }

        let edge = *max_uct_edge.unwrap();
        board.push_move(mcts.edges[edge].chess_move);

        // Case: picked a leaf node. Need to expand and return it.
        if max_uct_child == usize::MAX {
            let new_idx = MctsNode::new(board, edge, mcts);
            mcts.nodes[self.self_node_idx].children[child_idx] = (edge, Some(new_idx));
            board.pop_move();
            return
        }
        let next_node = mcts.nodes[max_uct_child].clone();
        next_node.select(board, mcts);
        board.pop_move();
    }
}

#[pyfunction]
pub fn mcts_search(board: &Board, c: f32, tau: f32, iterations: usize) -> Vec<(String, f32)> {
    let mut mcts = Mcts::new(c, board.to_move());

    let mut root = board.clone();
    let root_node = MctsNode::new(&mut root, usize::MAX, &mut mcts);
    for _ in 0..iterations {
        mcts.nodes[root_node].clone().select(&mut root, &mut mcts);
    }

    let move_scores = mcts.nodes[root_node].clone().children
        .into_iter()
        .map(|(edge_idx, _)| (mcts.edges[edge_idx].chess_move, mcts.edges[edge_idx].n))
        .collect::<Vec<_>>();

    let n_sum: usize = move_scores.iter().map(|(_, n)| n).sum();
    let denominator = (n_sum as f32).powf(1f32 / tau);

    move_scores.into_iter()
        .map(|(m, n)| (m.to_string(), (n as f32).powf(1f32 / tau) / denominator))
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::nnue::Nnue;

    use super::*;

    #[test]
    fn mcts_finds_mate_in_1() {
        let mut b = Board::from_fen("rnbqkbnr/pppp1ppp/4p3/8/5PP1/8/PPPPP2P/RNBQKBNR b KQkq - 0 2").unwrap();
        let n = Nnue::random();
        b.enable_nnue(n);

        let mut result = mcts_search(&b, 500f32, 0.5f32, 800);
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        assert_eq!("d8h4", result[0].0);

        let mut b = Board::from_fen("rnbqkbnr/ppppp2p/5p2/6p1/5P2/4P3/PPPP2PP/RNBQKBNR w KQkq - 0 3").unwrap();
        let n = Nnue::random();
        b.enable_nnue(n);

        let mut result = mcts_search(&b, 500f32, 0.5f32, 800);
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        assert_eq!("d1h5", result[0].0);
    }

    #[test]
    fn mcts_finds_mate_in_2() {
        let mut b = Board::from_fen("4r1k1/p4p1p/1p6/6B1/3P2n1/P4Q2/1P4P1/7K b - - 0 34").unwrap();
        let n = Nnue::random();
        b.enable_nnue(n);

        let mut result = mcts_search(&b, 500f32, 3f32, 8000);
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        assert_eq!("e8e1", result[0].0);

        let mut b = Board::from_fen("6r1/1r2kp1p/p1p1p1qp/4P3/8/6PP/P2Q1P2/3R2K1 w - - 0 26").unwrap();
        let n = Nnue::random();
        b.enable_nnue(n);

        let mut result = mcts_search(&b, 500f32, 3f32, 8000);
        result.sort_by(|a, b| b.1.total_cmp(&a.1));
        assert_eq!("d2d6", result[0].0);
    }
}