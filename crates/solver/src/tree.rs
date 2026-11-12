//! Bet abstraction and the flat public betting tree of one street.
//!
//! Amounts are in units of the pot at the start of the street (`P = 1`). `contrib[p]` is
//! what player `p` has put in on this street; the effective stack `stack` is what each
//! player has behind at the start of the street. Player 0 is out of position and acts first.
//!
//! Nodes are stored in breadth-first order, so every node's children are a contiguous index
//! range after it: a forward pass in index order visits parents before children and a
//! backward pass in reverse order visits children first.

/// Kind of a public node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// A player chooses an action.
    Decision,
    /// A player folded; the other wins the pot.
    Fold,
    /// Contributions are equal and the hand goes to showdown.
    Showdown,
    /// Reserved: a chance node dealing the next card (turn solving).
    Chance,
}

/// The action that led to a node (from its parent).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Edge {
    /// The root.
    Root,
    /// Check.
    Check,
    /// Bet to this street total (pot units).
    Bet(f32),
    /// Raise to this street total.
    Raise(f32),
    /// Bet or raise all-in.
    AllIn(f32),
    /// Call.
    Call,
    /// Fold.
    Fold,
}

/// Bet abstraction for one street.
#[derive(Clone, Debug, PartialEq)]
pub struct Abstraction {
    /// Bet sizes of the first player when nobody has bet (fractions of the pot).
    pub oop_bets: Vec<f32>,
    /// Bet sizes of the second player after a check.
    pub ip_bets: Vec<f32>,
    /// Raise sizes as fractions of the pot after calling (1.0 = pot-size raise).
    pub raises: Vec<f32>,
    /// Whether all-in is offered as a bet / raise.
    pub allin: bool,
    /// Maximum number of bets plus raises on the street.
    pub max_bets: u8,
    /// A bet or raise leaving less than this fraction of the pre-action stack behind is
    /// replaced by all-in (0 disables the rule).
    pub allin_threshold: f32,
}

impl Default for Abstraction {
    /// Bets {½ pot, pot, all-in}, raises {pot, all-in}, at most 3 bets/raises, ⅓ rule.
    fn default() -> Self {
        Abstraction {
            oop_bets: vec![0.5, 1.0],
            ip_bets: vec![0.5, 1.0],
            raises: vec![1.0],
            allin: true,
            max_bets: 3,
            allin_threshold: 1.0 / 3.0,
        }
    }
}

/// A flat public tree.
#[derive(Clone, Debug)]
pub struct Tree {
    /// Node kinds.
    pub kind: Vec<NodeKind>,
    /// Acting player (decision nodes) or the player who folded (fold nodes); 0 otherwise.
    pub player: Vec<u8>,
    /// Index of the first child (children are contiguous).
    pub first_child: Vec<u32>,
    /// Number of children (= actions) of a decision node; 0 for terminals.
    pub n_children: Vec<u8>,
    /// Street contributions `[p0, p1]` at the node, in pot units.
    pub contrib: Vec<[f32; 2]>,
    /// Parent index (`u32::MAX` for the root).
    pub parent: Vec<u32>,
    /// The action leading to each node.
    pub edge: Vec<Edge>,
    /// Per-player index of a decision node among that player's decisions.
    pub slot: Vec<u32>,
    /// Decision nodes per player.
    pub n_slots: [usize; 2],
    /// Largest number of actions at any decision.
    pub max_actions: usize,
    /// Effective stack at the start of the street (pot units).
    pub stack: f32,
}

struct Pending {
    parent: u32,
    edge: Edge,
    kind: NodeKind,
    player: u8,
    contrib: [f32; 2],
    bets: u8,
    checked: bool,
}

impl Tree {
    /// Builds the river tree for effective stack `stack` (= SPR) under `abs`.
    pub fn river(stack: f32, abs: &Abstraction) -> Tree {
        assert!(stack >= 0.0 && stack.is_finite(), "bad stack {stack}");
        let mut t = Tree {
            kind: vec![],
            player: vec![],
            first_child: vec![],
            n_children: vec![],
            contrib: vec![],
            parent: vec![],
            edge: vec![],
            slot: vec![],
            n_slots: [0, 0],
            max_actions: 1,
            stack,
        };
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(Pending {
            parent: u32::MAX,
            edge: Edge::Root,
            kind: NodeKind::Decision,
            player: 0,
            contrib: [0.0, 0.0],
            bets: 0,
            checked: false,
        });
        let mut state: Vec<(u8, bool)> = vec![]; // (bets, checked) per node, for expansion
                                                 // BFS: pushing a node's children right after it is popped keeps them contiguous.
        while let Some(p) = queue.pop_front() {
            let id = t.kind.len() as u32;
            t.kind.push(p.kind);
            t.player.push(p.player);
            t.first_child.push(0);
            t.n_children.push(0);
            t.contrib.push(p.contrib);
            t.parent.push(p.parent);
            t.edge.push(p.edge);
            t.slot.push(u32::MAX);
            state.push((p.bets, p.checked));
            if p.kind != NodeKind::Decision {
                continue;
            }
            t.slot[id as usize] = t.n_slots[p.player as usize] as u32;
            t.n_slots[p.player as usize] += 1;
            let children = Self::actions(stack, abs, p.player, p.contrib, p.bets, p.checked);
            t.max_actions = t.max_actions.max(children.len());
            t.n_children[id as usize] = children.len() as u8;
            // Children are assigned ids in queue order: first_child is fixed when the first
            // of them is popped. Record the parent; ids are patched below.
            for mut c in children {
                c.parent = id;
                queue.push_back(c);
            }
        }
        // Patch first_child: children of each node are contiguous, in order of appearance.
        let mut seen = vec![false; t.kind.len()];
        for i in 1..t.kind.len() {
            let par = t.parent[i] as usize;
            if !seen[par] {
                seen[par] = true;
                t.first_child[par] = i as u32;
            }
        }
        debug_assert!(t.check_contiguous());
        t
    }

    fn check_contiguous(&self) -> bool {
        (0..self.kind.len()).all(|i| {
            let f = self.first_child[i] as usize;
            (f..f + self.n_children[i] as usize).all(|c| self.parent[c] as usize == i)
        })
    }

    /// Children of a decision node with the given state.
    fn actions(
        stack: f32,
        abs: &Abstraction,
        p: u8,
        contrib: [f32; 2],
        bets: u8,
        checked: bool,
    ) -> Vec<Pending> {
        let o = 1 - p;
        let mine = contrib[p as usize];
        let theirs = contrib[o as usize];
        let mut out = Vec::new();
        let child = |edge, kind, player, contrib, bets, checked| Pending {
            parent: 0,
            edge,
            kind,
            player,
            contrib,
            bets,
            checked,
        };
        let eps = 1e-6f32;
        let can_raise = bets < abs.max_bets && theirs < stack - eps;
        // Street total of a sized bet/raise, with the all-in replacement rule.
        let sized = |to: f32| -> f32 {
            let pre = stack - mine;
            if to >= stack - eps || (stack - to) < abs.allin_threshold * pre {
                stack
            } else {
                to
            }
        };
        let push_sizes = |out: &mut Vec<Pending>, tos: Vec<f32>, raise: bool| {
            let mut tos: Vec<f32> = tos.into_iter().map(sized).collect();
            if abs.allin {
                tos.push(stack);
            }
            tos.sort_by(|a, b| a.partial_cmp(b).unwrap());
            tos.dedup_by(|a, b| (*a - *b).abs() < eps);
            for to in tos {
                if to <= theirs + eps {
                    continue; // not an increase (e.g. stack 0)
                }
                let mut c = contrib;
                c[p as usize] = to;
                let edge = if (to - stack).abs() < eps {
                    Edge::AllIn(to)
                } else if raise {
                    Edge::Raise(to)
                } else {
                    Edge::Bet(to)
                };
                out.push(child(edge, NodeKind::Decision, o, c, bets + 1, false));
            }
        };
        if (theirs - mine).abs() < eps {
            // Nobody to call: check or bet.
            if p == 1 || checked {
                out.push(child(
                    Edge::Check,
                    NodeKind::Showdown,
                    0,
                    contrib,
                    bets,
                    true,
                ));
            } else {
                out.push(child(
                    Edge::Check,
                    NodeKind::Decision,
                    o,
                    contrib,
                    bets,
                    true,
                ));
            }
            if can_raise {
                let pot = 1.0 + contrib[0] + contrib[1];
                let fr = if p == 0 { &abs.oop_bets } else { &abs.ip_bets };
                push_sizes(&mut out, fr.iter().map(|f| mine + f * pot).collect(), false);
            }
        } else {
            out.push(child(Edge::Fold, NodeKind::Fold, p, contrib, bets, checked));
            let mut c = contrib;
            c[p as usize] = theirs;
            out.push(child(Edge::Call, NodeKind::Showdown, 0, c, bets, checked));
            if can_raise {
                let pot_after_call = 1.0 + 2.0 * theirs;
                push_sizes(
                    &mut out,
                    abs.raises
                        .iter()
                        .map(|f| theirs + f * pot_after_call)
                        .collect(),
                    true,
                );
            }
        }
        out
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.kind.len()
    }

    /// True for an empty tree (never: there is always a root).
    pub fn is_empty(&self) -> bool {
        self.kind.is_empty()
    }

    /// Children of node `i`.
    pub fn children(&self, i: usize) -> std::ops::Range<usize> {
        let f = self.first_child[i] as usize;
        f..f + self.n_children[i] as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tree_shape() {
        let t = Tree::river(5.0, &Abstraction::default());
        assert!((20..=80).contains(&t.len()), "{} nodes", t.len());
        assert_eq!(t.kind[0], NodeKind::Decision);
        // Root: check, 0.5, 1, all-in.
        assert_eq!(t.n_children[0], 4);
        for i in 0..t.len() {
            if t.kind[i] == NodeKind::Showdown {
                assert!((t.contrib[i][0] - t.contrib[i][1]).abs() < 1e-6);
            }
            assert!(t.contrib[i][0] <= 5.0 + 1e-6 && t.contrib[i][1] <= 5.0 + 1e-6);
        }
        // Cap: no decision deeper than 3 bets/raises offers another raise.
    }

    #[test]
    fn allin_rule_and_small_stacks() {
        // SPR 0.6: a pot bet (1.0) exceeds the stack, a half-pot bet leaves 0.1 < 0.6/3
        // behind: both become all-in, so the root is check / all-in.
        let t = Tree::river(0.6, &Abstraction::default());
        assert_eq!(t.n_children[0], 2);
        // SPR 0: only check-check.
        let t = Tree::river(0.0, &Abstraction::default());
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn toy_tree() {
        let abs = Abstraction {
            oop_bets: vec![1.0],
            ip_bets: vec![],
            raises: vec![],
            allin: false,
            max_bets: 1,
            allin_threshold: 0.0,
        };
        let t = Tree::river(10.0, &abs);
        // root: check -> IP check -> showdown; bet -> fold / call.
        assert_eq!(t.len(), 6);
        assert_eq!(t.n_slots, [1, 2]);
    }
}
