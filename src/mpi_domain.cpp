// =============================================================================
//  src/mpi_domain.cpp
//
//  MPI implementation of the optional 2D Cartesian domain-decomposition layer
//  declared in include/mpi_domain.hpp.  This translation unit is compiled only
//  when the build is configured with -DENABLE_MPI; the serial build uses the
//  inline identity stubs in the header instead and never sees MPI.
// =============================================================================

#include "mpi_domain.hpp"

#ifdef ENABLE_MPI

#include <mpi.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "glm2d_common.hpp"  // idx2d

namespace {

constexpr int ND = NVAR;  // doubles carried per cell

// Pack the rectangular padded region [ci0, ci0+cw) x [cj0, cj0+ch) into buf.
void pack_states(
    const std::vector<State>& U, int nx_pad,
    int ci0, int cj0, int cw, int ch,
    std::vector<double>& buf
) {
    buf.resize(static_cast<std::size_t>(cw) * ch * ND);
    std::size_t k = 0;
    for (int j = cj0; j < cj0 + ch; ++j) {
        for (int i = ci0; i < ci0 + cw; ++i) {
            const State& s = U[idx2d(i, j, nx_pad)];
            for (int v = 0; v < ND; ++v) {
                buf[k++] = s[v];
            }
        }
    }
}

void unpack_states(
    std::vector<State>& U, int nx_pad,
    int ci0, int cj0, int cw, int ch,
    const std::vector<double>& buf
) {
    std::size_t k = 0;
    for (int j = cj0; j < cj0 + ch; ++j) {
        for (int i = ci0; i < ci0 + cw; ++i) {
            State& s = U[idx2d(i, j, nx_pad)];
            for (int v = 0; v < ND; ++v) {
                s[v] = buf[k++];
            }
        }
    }
}

// Sendrecv a packed State region to `dest` and receive a region of identical
// shape from `source`, then unpack into the destination rectangle.
void shift_states(
    std::vector<State>& U, int nx_pad,
    int send_ci0, int send_cj0, int cw, int ch,
    int recv_ci0, int recv_cj0,
    int dest, int source, MPI_Comm cart
) {
    std::vector<double> sbuf, rbuf;
    pack_states(U, nx_pad, send_ci0, send_cj0, cw, ch, sbuf);
    rbuf.resize(sbuf.size());

    MPI_Sendrecv(
        sbuf.data(), static_cast<int>(sbuf.size()), MPI_DOUBLE, dest, 7,
        rbuf.data(), static_cast<int>(rbuf.size()), MPI_DOUBLE, source, 7,
        cart, MPI_STATUS_IGNORE
    );

    unpack_states(U, nx_pad, recv_ci0, recv_cj0, cw, ch, rbuf);
}

// As shift_states but for a char mask, OR-combining the incoming values into
// the destination rectangle (so a face flagged on either side stays flagged).
void shift_mask_or(
    std::vector<char>& M, int nx_pad,
    int send_ci0, int send_cj0, int cw, int ch,
    int recv_ci0, int recv_cj0,
    int dest, int source, MPI_Comm cart
) {
    const std::size_t n = static_cast<std::size_t>(cw) * ch;
    std::vector<char> sbuf(n), rbuf(n);

    std::size_t k = 0;
    for (int j = send_cj0; j < send_cj0 + ch; ++j) {
        for (int i = send_ci0; i < send_ci0 + cw; ++i) {
            sbuf[k++] = M[idx2d(i, j, nx_pad)];
        }
    }

    MPI_Sendrecv(
        sbuf.data(), static_cast<int>(n), MPI_CHAR, dest, 8,
        rbuf.data(), static_cast<int>(n), MPI_CHAR, source, 8,
        cart, MPI_STATUS_IGNORE
    );

    k = 0;
    for (int j = recv_cj0; j < recv_cj0 + ch; ++j) {
        for (int i = recv_ci0; i < recv_ci0 + cw; ++i) {
            char& dst = M[idx2d(i, j, nx_pad)];
            dst = static_cast<char>(dst | rbuf[k++]);
        }
    }
}

}  // namespace

// -----------------------------------------------------------------------------

MPIDomain make_domain(int nx_g, int ny_g, int ng) {
    MPIDomain d;
    d.nx_g = nx_g;
    d.ny_g = ny_g;
    d.ng = ng;

    MPI_Comm_size(MPI_COMM_WORLD, &d.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);

    // Factor the world size into a 2D process grid.
    int dims[2] = {0, 0};
    MPI_Dims_create(d.size, 2, dims);
    d.dims[0] = dims[0];
    d.dims[1] = dims[1];

    if (nx_g % dims[0] != 0 || ny_g % dims[1] != 0) {
        throw std::invalid_argument(
            "make_domain: global grid " + std::to_string(nx_g) + "x"
          + std::to_string(ny_g) + " not divisible by process grid "
          + std::to_string(dims[0]) + "x" + std::to_string(dims[1])
          + " (choose nx/ny or rank count so each rank gets equal blocks)");
    }

    const int periods[2] = {1, 1};  // periodic in both directions
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/0, &d.cart);
    MPI_Cart_coords(d.cart, d.rank, 2, d.coords);

    // Neighbours (periodic, so always valid even for a 1-wide dimension, where
    // the neighbour is this rank itself and the halo exchange degenerates to the
    // serial periodic copy).
    MPI_Cart_shift(d.cart, 0, 1, &d.nbr_left, &d.nbr_right);
    MPI_Cart_shift(d.cart, 1, 1, &d.nbr_down, &d.nbr_up);

    d.nx_loc = nx_g / dims[0];
    d.ny_loc = ny_g / dims[1];
    d.i0 = d.coords[0] * d.nx_loc;
    d.j0 = d.coords[1] * d.ny_loc;

    if (d.nx_loc < ng || d.ny_loc < ng) {
        throw std::invalid_argument(
            "make_domain: per-rank block smaller than ghost width; use fewer "
            "ranks or a larger grid");
    }

    d.active = true;
    return d;
}

void exchange_halos(std::vector<State>& U, const MPIDomain* d) {
    if (!d || !d->active) {
        return;
    }

    const int ng = d->ng;
    const int nx_pad = d->nx_pad();
    const int nx_loc = d->nx_loc;
    const int ny_loc = d->ny_loc;
    const int ny_pad = d->ny_pad();

    // Phase 1: vertical exchange over interior columns only.
    //   - send my top interior rows    -> up neighbour   (fills its bottom ghost)
    //   - receive into my bottom ghost  <- down neighbour
    shift_states(U, nx_pad,
                 /*send*/ ng, ny_loc,        nx_loc, ng,   // top interior rows
                 /*recv*/ ng, 0,                            // bottom ghost
                 d->nbr_up, d->nbr_down, d->cart);
    //   - send my bottom interior rows -> down neighbour (fills its top ghost)
    //   - receive into my top ghost     <- up neighbour
    shift_states(U, nx_pad,
                 /*send*/ ng, ng,            nx_loc, ng,   // bottom interior rows
                 /*recv*/ ng, ng + ny_loc,                  // top ghost
                 d->nbr_down, d->nbr_up, d->cart);

    // Phase 2: horizontal exchange over the full padded height, so the corner
    // ghosts (just filled in phase 1) propagate diagonally.
    shift_states(U, nx_pad,
                 /*send*/ nx_loc, 0,         ng, ny_pad,   // right interior cols
                 /*recv*/ 0, 0,                             // left ghost
                 d->nbr_right, d->nbr_left, d->cart);
    shift_states(U, nx_pad,
                 /*send*/ ng, 0,             ng, ny_pad,   // left interior cols
                 /*recv*/ ng + nx_loc, 0,                   // right ghost
                 d->nbr_left, d->nbr_right, d->cart);
}

void exchange_halo_mask_or(std::vector<char>& M, const MPIDomain* d) {
    if (!d || !d->active) {
        return;
    }

    const int ng = d->ng;
    const int nx_pad = d->nx_pad();
    const int nx_loc = d->nx_loc;
    const int ny_loc = d->ny_loc;
    const int ny_pad = d->ny_pad();

    shift_mask_or(M, nx_pad, ng, ny_loc, nx_loc, ng, ng, 0,
                  d->nbr_up, d->nbr_down, d->cart);
    shift_mask_or(M, nx_pad, ng, ng,     nx_loc, ng, ng, ng + ny_loc,
                  d->nbr_down, d->nbr_up, d->cart);
    shift_mask_or(M, nx_pad, nx_loc, 0,  ng, ny_pad, 0, 0,
                  d->nbr_right, d->nbr_left, d->cart);
    shift_mask_or(M, nx_pad, ng, 0,      ng, ny_pad, ng + nx_loc, 0,
                  d->nbr_left, d->nbr_right, d->cart);
}

double global_max(double local, const MPIDomain* d) {
    if (!d || !d->active) return local;
    double out = local;
    MPI_Allreduce(&local, &out, 1, MPI_DOUBLE, MPI_MAX, d->cart);
    return out;
}

double global_min(double local, const MPIDomain* d) {
    if (!d || !d->active) return local;
    double out = local;
    MPI_Allreduce(&local, &out, 1, MPI_DOUBLE, MPI_MIN, d->cart);
    return out;
}

double global_sum(double local, const MPIDomain* d) {
    if (!d || !d->active) return local;
    double out = local;
    MPI_Allreduce(&local, &out, 1, MPI_DOUBLE, MPI_SUM, d->cart);
    return out;
}

long long global_sum(long long local, const MPIDomain* d) {
    if (!d || !d->active) return local;
    long long out = local;
    MPI_Allreduce(&local, &out, 1, MPI_LONG_LONG, MPI_SUM, d->cart);
    return out;
}

int global_lor(int local_flag, const MPIDomain* d) {
    if (!d || !d->active) return local_flag;
    int out = local_flag;
    MPI_Allreduce(&local_flag, &out, 1, MPI_INT, MPI_LOR, d->cart);
    return out;
}

bool global_all(bool local_true, const MPIDomain* d) {
    if (!d || !d->active) return local_true;
    int in = local_true ? 1 : 0;
    int out = in;
    MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_LAND, d->cart);
    return out != 0;
}

std::vector<State> gather_to_root(
    const std::vector<State>& U_padded,
    const MPIDomain* d
) {
    if (!d || !d->active) {
        return U_padded;
    }

    const int ng = d->ng;
    const int nx_pad = d->nx_pad();
    const int nx_loc = d->nx_loc;
    const int ny_loc = d->ny_loc;
    const int block = nx_loc * ny_loc;

    // Pack this rank's interior block (excluding ghosts) in (i fast) order.
    std::vector<double> sbuf;
    pack_states(U_padded, nx_pad, ng, ng, nx_loc, ny_loc, sbuf);

    std::vector<double> rbuf;
    if (d->rank == 0) {
        rbuf.resize(static_cast<std::size_t>(block) * ND * d->size);
    }

    MPI_Gather(
        sbuf.data(), block * ND, MPI_DOUBLE,
        rbuf.data(), block * ND, MPI_DOUBLE,
        0, d->cart
    );

    if (d->rank != 0) {
        return {};
    }

    std::vector<State> global(static_cast<std::size_t>(d->nx_g) * d->ny_g);
    for (int r = 0; r < d->size; ++r) {
        int rc[2];
        MPI_Cart_coords(d->cart, r, 2, rc);
        const int gi0 = rc[0] * nx_loc;
        const int gj0 = rc[1] * ny_loc;
        std::size_t k = static_cast<std::size_t>(r) * block * ND;
        for (int j = 0; j < ny_loc; ++j) {
            for (int i = 0; i < nx_loc; ++i) {
                State& s = global[idx2d(gi0 + i, gj0 + j, d->nx_g)];
                for (int v = 0; v < ND; ++v) {
                    s[v] = rbuf[k++];
                }
            }
        }
    }
    return global;
}

#endif  // ENABLE_MPI
