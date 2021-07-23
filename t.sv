module mult
  #(parameter WIDTH = 64,
    parameter INT_WIDTH = 32,
    parameter FRAC_WIDTH = 32)
   (input logic [WIDTH-1:0]  left,
    input logic [WIDTH-1:0]  right,
    input logic              go,
    input logic              clk,
    output logic [WIDTH-1:0] out,
    output logic             done);
   logic [WIDTH-1:0]         rtmp;
   logic [WIDTH-1:0]         ltmp;
   logic [(WIDTH << 1) - 1:0] out_tmp;
   reg                        done_buf[1:0];
   always_ff @(posedge clk) begin
      if (go) begin
         rtmp <= right;
         ltmp <= left;
         out_tmp <= ltmp * rtmp;
         out <= out_tmp[(WIDTH << 1) - INT_WIDTH - 1 : WIDTH - INT_WIDTH];

         done <= done_buf[1];
         done_buf[0] <= 1'b1;
         done_buf[1] <= done_buf[0];
      end else begin
         rtmp <= 0;
         ltmp <= 0;
         out_tmp <= 0;
         out <= 0;

         done <= 0;
         done_buf[0] <= 0;
         done_buf[1] <= 0;
      end
   end
endmodule

module add
  #(parameter WIDTH = 64)
   (input signed [WIDTH-1:0]  left,
    input signed [WIDTH-1:0]  right,
    input                     clk,
    input                     go,
    output signed [WIDTH-1:0] out,
    output logic              done);
   logic [WIDTH-1:0]           sum;
   logic overflow;
   logic underflow;
   always_ff @(posedge clk) begin
      if (go)
        done <= 1'b1;
      else
        done <= 1'b0;
   end
   assign sum = $signed(left + right);
   assign overflow = (~left[WIDTH-1]) & (~right[WIDTH-1]) & sum[WIDTH-1];
   assign underflow = left[WIDTH-1] & right[WIDTH-1] & ~sum[WIDTH-1];
   assign out = overflow ? {1'b0,{WIDTH-1{1'b1}}}
                : underflow ? {WIDTH{1'b1}}
                : sum;
endmodule

module sub
  #(parameter WIDTH = 64)
   (input signed [WIDTH-1:0]  left,
    input signed [WIDTH-1:0]  right,
    input                     go,
    input                     clk,
    output signed [WIDTH-1:0] out,
    output logic              done);
   logic [WIDTH-1:0]           sum;
   logic                       overflow;
   logic                       underflow;

   always_ff @(posedge clk) begin
      if (go)
        done <= 1'b1;
      else
        done <= 1'b0;
   end

   assign sum = $signed(left - right);
   assign overflow = (~left[WIDTH-1]) & (right[WIDTH-1]) & sum[WIDTH-1];
   assign underflow = left[WIDTH-1] & (~right[WIDTH-1]) & ~sum[WIDTH-1];
   assign out = overflow ? {1'b0,{WIDTH-1{1'b1}}}
                : underflow ? {WIDTH{1'b1}}
                : sum;
endmodule
/// =================== Unsigned, Fixed Point =========================
module std_fp_add #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out
);
  assign out = left + right;
endmodule

module std_fp_sub #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out
);
  assign out = left - right;
endmodule

module std_fp_mult_pipe #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    input  logic             go,
    input  logic             clk,
    output logic [WIDTH-1:0] out,
    output logic             done
);
  logic [WIDTH-1:0]          rtmp;
  logic [WIDTH-1:0]          ltmp;
  logic [(WIDTH << 1) - 1:0] out_tmp;
  reg done_buf[1:0];
  always_ff @(posedge clk) begin
    if (go) begin
      rtmp <= right;
      ltmp <= left;
      out_tmp <= ltmp * rtmp;
      out <= out_tmp[(WIDTH << 1) - INT_WIDTH - 1 : WIDTH - INT_WIDTH];

      done <= done_buf[1];
      done_buf[0] <= 1'b1;
      done_buf[1] <= done_buf[0];
    end else begin
      rtmp <= 0;
      ltmp <= 0;
      out_tmp <= 0;
      out <= 0;

      done <= 0;
      done_buf[0] <= 0;
      done_buf[1] <= 0;
    end
  end
endmodule

/* verilator lint_off WIDTH */
module std_fp_div_pipe #(
  parameter WIDTH = 32,
  parameter INT_WIDTH = 16,
  parameter FRAC_WIDTH = 16
) (
    input  logic             go,
    input  logic             clk,
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out_remainder,
    output logic [WIDTH-1:0] out_quotient,
    output logic             done
);
    localparam ITERATIONS = WIDTH + FRAC_WIDTH;

    logic [WIDTH-1:0] quotient, quotient_next;
    logic [WIDTH:0] acc, acc_next;
    logic [$clog2(ITERATIONS)-1:0] idx;
    logic start, running, finished;

    assign start = go && !running;
    assign finished = running && (idx == ITERATIONS - 1);

    always_comb begin
      if (acc >= {1'b0, right}) begin
        acc_next = acc - right;
        {acc_next, quotient_next} = {acc_next[WIDTH-1:0], quotient, 1'b1};
      end else begin
        {acc_next, quotient_next} = {acc, quotient} << 1;
      end
    end

    always_ff @(posedge clk) begin
      if (!go) begin
        running <= 0;
        done <= 0;
        out_remainder <= 0;
        out_quotient <= 0;
      end else if (start && left == 0) begin
        out_remainder <= 0;
        out_quotient <= 0;
        done <= 1;
      end

      if (start) begin
        running <= 1;
        done <= 0;
        idx <= 0;
        {acc, quotient} <= {{WIDTH{1'b0}}, left, 1'b0};
        out_quotient <= 0;
        out_remainder <= left;
      end else if (finished) begin
        running <= 0;
        done <= 1;
        out_quotient <= quotient_next;
      end else begin
        idx <= idx + 1;
        acc <= acc_next;
        quotient <= quotient_next;
        if (right <= out_remainder) begin
          out_remainder <= out_remainder - right;
        end
      end
    end
endmodule

module std_fp_gt #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    output logic             out
);
  assign out = left > right;
endmodule

module std_fp_add_dwidth #(
    parameter WIDTH1 = 32,
    parameter WIDTH2 = 32,
    parameter INT_WIDTH1 = 16,
    parameter FRAC_WIDTH1 = 16,
    parameter INT_WIDTH2 = 12,
    parameter FRAC_WIDTH2 = 20,
    parameter OUT_WIDTH = 36
) (
    input  logic [   WIDTH1-1:0] left,
    input  logic [   WIDTH2-1:0] right,
    output logic [OUT_WIDTH-1:0] out
);

  localparam BIG_INT = (INT_WIDTH1 >= INT_WIDTH2) ? INT_WIDTH1 : INT_WIDTH2;
  localparam BIG_FRACT = (FRAC_WIDTH1 >= FRAC_WIDTH2) ? FRAC_WIDTH1 : FRAC_WIDTH2;

  if (BIG_INT + BIG_FRACT != OUT_WIDTH)
    $error("std_fp_add_dwidth: Given output width not equal to computed output width");

  logic [INT_WIDTH1-1:0] left_int;
  logic [INT_WIDTH2-1:0] right_int;
  logic [FRAC_WIDTH1-1:0] left_fract;
  logic [FRAC_WIDTH2-1:0] right_fract;

  logic [BIG_INT-1:0] mod_right_int;
  logic [BIG_FRACT-1:0] mod_left_fract;

  logic [BIG_INT-1:0] whole_int;
  logic [BIG_FRACT-1:0] whole_fract;

  assign {left_int, left_fract} = left;
  assign {right_int, right_fract} = right;

  assign mod_left_fract = left_fract * (2 ** (FRAC_WIDTH2 - FRAC_WIDTH1));

  always_comb begin
    if ((mod_left_fract + right_fract) >= 2 ** FRAC_WIDTH2) begin
      whole_int = left_int + right_int + 1;
      whole_fract = mod_left_fract + right_fract - 2 ** FRAC_WIDTH2;
    end else begin
      whole_int = left_int + right_int;
      whole_fract = mod_left_fract + right_fract;
    end
  end

  assign out = {whole_int, whole_fract};
endmodule

/// =================== Signed, Fixed Point =========================
module std_fp_sadd #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = $signed(left + right);
endmodule

module std_fp_ssub #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);

  assign out = $signed(left - right);
endmodule

module std_fp_smult_pipe #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  signed       [WIDTH-1:0] left,
    input  signed       [WIDTH-1:0] right,
    input  logic                    go,
    input  logic                    clk,
    output logic signed [WIDTH-1:0] out,
    output logic                    done
);
  logic signed [WIDTH-1:0] ltmp;
  logic signed [WIDTH-1:0] rtmp;
  logic signed [(WIDTH << 1) - 1:0] out_tmp;
  reg done_buf[1:0];
  always_ff @(posedge clk) begin
    if (go) begin
      ltmp <= left;
      rtmp <= right;
      // Sign extend by the first bit for the operands.
      out_tmp <= $signed(
                   { {WIDTH{ltmp[WIDTH-1]}}, ltmp} *
                   { {WIDTH{rtmp[WIDTH-1]}}, rtmp}
                 );
      out <= out_tmp[(WIDTH << 1) - INT_WIDTH - 1: WIDTH - INT_WIDTH];

      done <= done_buf[1];
      done_buf[0] <= 1'b1;
      done_buf[1] <= done_buf[0];
    end else begin
      rtmp <= 0;
      ltmp <= 0;
      out_tmp <= 0;
      out <= 0;

      done <= 0;
      done_buf[0] <= 0;
      done_buf[1] <= 0;
    end
  end
endmodule

module std_fp_sdiv_pipe #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input                     clk,
    input                     go,
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out_quotient,
    output signed [WIDTH-1:0] out_remainder,
    output logic              done
);

  logic signed [WIDTH-1:0] left_abs;
  logic signed [WIDTH-1:0] right_abs;
  logic signed [WIDTH-1:0] comp_out_q;
  logic signed [WIDTH-1:0] comp_out_r;

  assign right_abs = right[WIDTH-1] ? -right : right;
  assign left_abs = left[WIDTH-1] ? -left : left;
  assign out_quotient = left[WIDTH-1] ^ right[WIDTH-1] ? -comp_out_q : comp_out_q;
  assign out_remainder = (left[WIDTH-1] && comp_out_r) ? $signed(right - comp_out_r) : comp_out_r;

  std_fp_div_pipe #(
    .WIDTH(WIDTH),
    .INT_WIDTH(INT_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH)
  ) comp (
    .clk(clk),
    .done(done),
    .go(go),
    .left(left_abs),
    .right(right_abs),
    .out_quotient(comp_out_q),
    .out_remainder(comp_out_r)
  );
endmodule

module std_fp_sadd_dwidth #(
    parameter WIDTH1 = 32,
    parameter WIDTH2 = 32,
    parameter INT_WIDTH1 = 16,
    parameter FRAC_WIDTH1 = 16,
    parameter INT_WIDTH2 = 12,
    parameter FRAC_WIDTH2 = 20,
    parameter OUT_WIDTH = 36
) (
    input  logic [   WIDTH1-1:0] left,
    input  logic [   WIDTH2-1:0] right,
    output logic [OUT_WIDTH-1:0] out
);

  logic signed [INT_WIDTH1-1:0] left_int;
  logic signed [INT_WIDTH2-1:0] right_int;
  logic [FRAC_WIDTH1-1:0] left_fract;
  logic [FRAC_WIDTH2-1:0] right_fract;

  localparam BIG_INT = (INT_WIDTH1 >= INT_WIDTH2) ? INT_WIDTH1 : INT_WIDTH2;
  localparam BIG_FRACT = (FRAC_WIDTH1 >= FRAC_WIDTH2) ? FRAC_WIDTH1 : FRAC_WIDTH2;

  logic [BIG_INT-1:0] mod_right_int;
  logic [BIG_FRACT-1:0] mod_left_fract;

  logic [BIG_INT-1:0] whole_int;
  logic [BIG_FRACT-1:0] whole_fract;

  assign {left_int, left_fract} = left;
  assign {right_int, right_fract} = right;

  assign mod_left_fract = left_fract * (2 ** (FRAC_WIDTH2 - FRAC_WIDTH1));

  always_comb begin
    if ((mod_left_fract + right_fract) >= 2 ** FRAC_WIDTH2) begin
      whole_int = $signed(left_int + right_int + 1);
      whole_fract = mod_left_fract + right_fract - 2 ** FRAC_WIDTH2;
    end else begin
      whole_int = $signed(left_int + right_int);
      whole_fract = mod_left_fract + right_fract;
    end
  end

  assign out = {whole_int, whole_fract};
endmodule

module std_fp_sgt #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
    input  logic signed [WIDTH-1:0] left,
    input  logic signed [WIDTH-1:0] right,
    output logic signed             out
);
  assign out = $signed(left > right);
endmodule

module std_fp_slt #(
    parameter WIDTH = 32,
    parameter INT_WIDTH = 16,
    parameter FRAC_WIDTH = 16
) (
   input logic signed [WIDTH-1:0] left,
   input logic signed [WIDTH-1:0] right,
   output logic signed            out
);
  assign out = $signed(left < right);
endmodule

/// =================== Unsigned, Bitnum =========================
module std_mult_pipe #(
    parameter WIDTH = 32
) (
    input  logic [WIDTH-1:0] left,
    input  logic [WIDTH-1:0] right,
    input  logic             go,
    input  logic             clk,
    output logic [WIDTH-1:0] out,
    output logic             done
);
  std_fp_mult_pipe #(
    .WIDTH(WIDTH),
    .INT_WIDTH(WIDTH),
    .FRAC_WIDTH(0)
  ) comp (
    .clk(clk),
    .done(done),
    .go(go),
    .left(left),
    .right(right),
    .out(out)
  );
endmodule

module std_div_pipe #(
    parameter WIDTH = 32
) (
    input                    clk,
    input                    go,
    input        [WIDTH-1:0] left,
    input        [WIDTH-1:0] right,
    output logic [WIDTH-1:0] out_remainder,
    output logic [WIDTH-1:0] out_quotient,
    output logic             done
);

  logic [WIDTH-1:0] dividend;
  logic [(WIDTH-1)*2:0] divisor;
  logic [WIDTH-1:0] quotient;
  logic [WIDTH-1:0] quotient_msk;
  logic start, running, finished;

  assign start = go && !running;
  assign finished = !quotient_msk && running;

  always_ff @(posedge clk) begin
    if (!go) begin
      running <= 0;
      done <= 0;
      out_remainder <= 0;
      out_quotient <= 0;
    end else if (start && left == 0) begin
      out_remainder <= 0;
      out_quotient <= 0;
      done <= 1;
    end

    if (start) begin
      running <= 1;
      dividend <= left;
      divisor <= right << WIDTH - 1;
      quotient <= 0;
      quotient_msk <= 1 << WIDTH - 1;
    end else if (finished) begin
      running <= 0;
      done <= 1;
      out_remainder <= dividend;
      out_quotient <= quotient;
    end else begin
      if (divisor <= dividend) begin
        dividend <= dividend - divisor;
        quotient <= quotient | quotient_msk;
      end
      divisor <= divisor >> 1;
      quotient_msk <= quotient_msk >> 1;
    end
  end

  `ifdef VERILATOR
    // Simulation self test against unsynthesizable implementation.
    always @(posedge clk) begin
      if (finished && dividend != $unsigned(left % right))
        $error(
          "\nstd_div_pipe (Remainder): Computed and golden outputs do not match!\n",
          "left: %0d", $unsigned(left),
          "  right: %0d\n", $unsigned(right),
          "expected: %0d", $unsigned(left % right),
          "  computed: %0d", $unsigned(dividend)
        );
      if (finished && quotient != $unsigned(left / right))
        $error(
          "\nstd_div_pipe (Quotient): Computed and golden outputs do not match!\n",
          "left: %0d", $unsigned(left),
          "  right: %0d\n", $unsigned(right),
          "expected: %0d", $unsigned(left / right),
          "  computed: %0d", $unsigned(quotient)
        );
    end
  `endif
endmodule

/// =================== Signed, Bitnum =========================
module std_sadd #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = $signed(left + right);
endmodule

module std_ssub #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = $signed(left - right);
endmodule

module std_smult_pipe #(
    parameter WIDTH = 32
) (
    input  logic                    go,
    input  logic                    clk,
    input  signed       [WIDTH-1:0] left,
    input  signed       [WIDTH-1:0] right,
    output logic signed [WIDTH-1:0] out,
    output logic                    done
);
  std_fp_smult_pipe #(
    .WIDTH(WIDTH),
    .INT_WIDTH(WIDTH),
    .FRAC_WIDTH(0)
  ) comp (
    .clk(clk),
    .done(done),
    .go(go),
    .left(left),
    .right(right),
    .out(out)
  );
endmodule

/* verilator lint_off WIDTH */
module std_sdiv_pipe #(
    parameter WIDTH = 32
) (
    input                     clk,
    input                     go,
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out_quotient,
    output signed [WIDTH-1:0] out_remainder,
    output logic              done
);

  logic signed [WIDTH-1:0] left_abs, right_abs, comp_out_q, comp_out_r;
  logic different_signs;

  assign right_abs = right[WIDTH-1] ? -right : right;
  assign left_abs = left[WIDTH-1] ? -left : left;
  assign different_signs = left[WIDTH-1] ^ right[WIDTH-1];
  assign out_quotient = different_signs ? -comp_out_q : comp_out_q;
  assign out_remainder = (left[WIDTH-1] && comp_out_r) ? $signed(right - comp_out_r) : comp_out_r;

  std_div_pipe #(
    .WIDTH(WIDTH)
  ) comp (
    .clk(clk),
    .done(done),
    .go(go),
    .left(left_abs),
    .right(right_abs),
    .out_quotient(comp_out_q),
    .out_remainder(comp_out_r)
  );

  `ifdef VERILATOR
    // Simulation self test against unsynthesizable implementation.
    always @(posedge clk) begin
      if (done && out_quotient != $signed(left / right))
        $error(
          "\nstd_sdiv_pipe (Quotient): Computed and golden outputs do not match!\n",
          "left: %0d", left,
          "  right: %0d\n", right,
          "expected: %0d", $signed(left / right),
          "  computed: %0d", $signed(out_quotient)
        );
      if (done && out_remainder != $signed(((left % right) + right) % right))
        $error(
          "\nstd_sdiv_pipe (Remainder): Computed and golden outputs do not match!\n",
          "left: %0d", left,
          "  right: %0d\n", right,
          "expected: %0d", $signed(((left % right) + right) % right),
          "  computed: %0d", $signed(out_remainder)
        );
    end
  `endif
endmodule

module std_sgt #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left > right);
endmodule

module std_slt #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left < right);
endmodule

module std_seq #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left == right);
endmodule

module std_sneq #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left != right);
endmodule

module std_sge #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left >= right);
endmodule

module std_sle #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed             out
);
  assign out = $signed(left <= right);
endmodule

module std_slsh #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = left <<< right;
endmodule

module std_srsh #(
    parameter WIDTH = 32
) (
    input  signed [WIDTH-1:0] left,
    input  signed [WIDTH-1:0] right,
    output signed [WIDTH-1:0] out
);
  assign out = left >>> right;
endmodule
/**
 * Core primitives for Calyx.
 * Implements core primitives used by the compiler.
 *
 * Conventions:
 * - All parameter names must be SNAKE_CASE and all caps.
 * - Port names must be snake_case, no caps.
 */
`default_nettype none

module std_const #(
    parameter WIDTH = 32,
    parameter VALUE = 0
) (
   output logic [WIDTH - 1:0] out
);
  assign out = VALUE;
endmodule

module std_slice #(
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = 32
) (
   input wire                   logic [ IN_WIDTH-1:0] in,
   output logic [OUT_WIDTH-1:0] out
);
  assign out = in[OUT_WIDTH-1:0];

  `ifdef VERILATOR
    always_comb begin
      if (IN_WIDTH < OUT_WIDTH)
        $error(
          "std_slice: Input width less than output width\n",
          "IN_WIDTH: %0d", IN_WIDTH,
          "OUT_WIDTH: %0d", OUT_WIDTH
        );
    end
  `endif
endmodule

module std_pad #(
    parameter IN_WIDTH  = 32,
    parameter OUT_WIDTH = 32
) (
   input wire logic [IN_WIDTH-1:0]  in,
   output logic     [OUT_WIDTH-1:0] out
);
  localparam EXTEND = OUT_WIDTH - IN_WIDTH;
  assign out = { {EXTEND {1'b0}}, in};

  `ifdef VERILATOR
    always_comb begin
      if (IN_WIDTH > OUT_WIDTH)
        $error(
          "std_pad: Output width less than input width\n",
          "IN_WIDTH: %0d", IN_WIDTH,
          "OUT_WIDTH: %0d", OUT_WIDTH
        );
    end
  `endif
endmodule

module std_not #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] in,
   output logic [WIDTH-1:0] out
);
  assign out = ~in;
endmodule

module std_and #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left & right;
endmodule

module std_or #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left | right;
endmodule

module std_xor #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left ^ right;
endmodule

module std_add #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left + right;
endmodule

module std_sub #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left - right;
endmodule

module std_gt #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left > right;
endmodule

module std_lt #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left < right;
endmodule

module std_eq #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left == right;
endmodule

module std_neq #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left != right;
endmodule

module std_ge #(
    parameter WIDTH = 32
) (
    input wire   logic [WIDTH-1:0] left,
    input wire   logic [WIDTH-1:0] right,
    output logic out
);
  assign out = left >= right;
endmodule

module std_le #(
    parameter WIDTH = 32
) (
   input wire   logic [WIDTH-1:0] left,
   input wire   logic [WIDTH-1:0] right,
   output logic out
);
  assign out = left <= right;
endmodule

module std_lsh #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left << right;
endmodule

module std_rsh #(
    parameter WIDTH = 32
) (
   input wire               logic [WIDTH-1:0] left,
   input wire               logic [WIDTH-1:0] right,
   output logic [WIDTH-1:0] out
);
  assign out = left >> right;
endmodule

/// this primitive is intended to be used
/// for lowering purposes (not in source programs)
module std_mux #(
    parameter WIDTH = 32
) (
   input wire               logic cond,
   input wire               logic [WIDTH-1:0] tru,
   input wire               logic [WIDTH-1:0] fal,
   output logic [WIDTH-1:0] out
);
  assign out = cond ? tru : fal;
endmodule

/// Memories
module std_reg #(
    parameter WIDTH = 32
) (
   input wire [ WIDTH-1:0]    in,
   input wire                 write_en,
   input wire                 clk,
   input wire                 reset,
    // output
   output logic [WIDTH - 1:0] out,
   output logic               done
);

  always_ff @(posedge clk) begin
    if (reset) begin
       out <= 0;
       done <= 0;
    end else if (write_en) begin
      out <= in;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

module std_mem_d1 #(
    parameter WIDTH = 32,
    parameter SIZE = 16,
    parameter IDX_SIZE = 4
) (
   input wire                logic [IDX_SIZE-1:0] addr0,
   input wire                logic [ WIDTH-1:0] write_data,
   input wire                logic write_en,
   input wire                logic clk,
   output logic [ WIDTH-1:0] read_data,
   output logic              done
);

  logic [WIDTH-1:0] mem[SIZE-1:0];

  /* verilator lint_off WIDTH */
  assign read_data = mem[addr0];
  always_ff @(posedge clk) begin
    if (write_en) begin
      mem[addr0] <= write_data;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

module std_mem_d2 #(
    parameter WIDTH = 32,
    parameter D0_SIZE = 16,
    parameter D1_SIZE = 16,
    parameter D0_IDX_SIZE = 4,
    parameter D1_IDX_SIZE = 4
) (
   input wire                logic [D0_IDX_SIZE-1:0] addr0,
   input wire                logic [D1_IDX_SIZE-1:0] addr1,
   input wire                logic [ WIDTH-1:0] write_data,
   input wire                logic write_en,
   input wire                logic clk,
   output logic [ WIDTH-1:0] read_data,
   output logic              done
);

  /* verilator lint_off WIDTH */
  logic [WIDTH-1:0] mem[D0_SIZE-1:0][D1_SIZE-1:0];

  assign read_data = mem[addr0][addr1];
  always_ff @(posedge clk) begin
    if (write_en) begin
      mem[addr0][addr1] <= write_data;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

module std_mem_d3 #(
    parameter WIDTH = 32,
    parameter D0_SIZE = 16,
    parameter D1_SIZE = 16,
    parameter D2_SIZE = 16,
    parameter D0_IDX_SIZE = 4,
    parameter D1_IDX_SIZE = 4,
    parameter D2_IDX_SIZE = 4
) (
   input wire                logic [D0_IDX_SIZE-1:0] addr0,
   input wire                logic [D1_IDX_SIZE-1:0] addr1,
   input wire                logic [D2_IDX_SIZE-1:0] addr2,
   input wire                logic [ WIDTH-1:0] write_data,
   input wire                logic write_en,
   input wire                logic clk,
   output logic [ WIDTH-1:0] read_data,
   output logic              done
);

  /* verilator lint_off WIDTH */
  logic [WIDTH-1:0] mem[D0_SIZE-1:0][D1_SIZE-1:0][D2_SIZE-1:0];

  assign read_data = mem[addr0][addr1][addr2];
  always_ff @(posedge clk) begin
    if (write_en) begin
      mem[addr0][addr1][addr2] <= write_data;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

module std_mem_d4 #(
    parameter WIDTH = 32,
    parameter D0_SIZE = 16,
    parameter D1_SIZE = 16,
    parameter D2_SIZE = 16,
    parameter D3_SIZE = 16,
    parameter D0_IDX_SIZE = 4,
    parameter D1_IDX_SIZE = 4,
    parameter D2_IDX_SIZE = 4,
    parameter D3_IDX_SIZE = 4
) (
   input wire                logic [D0_IDX_SIZE-1:0] addr0,
   input wire                logic [D1_IDX_SIZE-1:0] addr1,
   input wire                logic [D2_IDX_SIZE-1:0] addr2,
   input wire                logic [D3_IDX_SIZE-1:0] addr3,
   input wire                logic [ WIDTH-1:0] write_data,
   input wire                logic write_en,
   input wire                logic clk,
   output logic [ WIDTH-1:0] read_data,
   output logic              done
);

  /* verilator lint_off WIDTH */
  logic [WIDTH-1:0] mem[D0_SIZE-1:0][D1_SIZE-1:0][D2_SIZE-1:0][D3_SIZE-1:0];

  assign read_data = mem[addr0][addr1][addr2][addr3];
  always_ff @(posedge clk) begin
    if (write_en) begin
      mem[addr0][addr1][addr2][addr3] <= write_data;
      done <= 1'd1;
    end else done <= 1'd0;
  end
endmodule

`default_nettype wire
module main (
    input logic go,
    input logic clk,
    input logic reset,
    output logic done
);
    import "DPI-C" function string futil_getenv (input string env_var);
    string DATA;
    initial begin
        DATA = futil_getenv("DATA");
        $fdisplay(2, "DATA (path to meminit files): %s", DATA);
        $readmemh({DATA, "/int_c_img0.dat"}, int_c_img0.mem);
        $readmemh({DATA, "/int_c_real0.dat"}, int_c_real0.mem);
        $readmemh({DATA, "/int_outputs0.dat"}, int_outputs0.mem);
    end
    final begin
        $writememh({DATA, "/int_c_img0.out"}, int_c_img0.mem);
        $writememh({DATA, "/int_c_real0.out"}, int_c_real0.mem);
        $writememh({DATA, "/int_outputs0.out"}, int_outputs0.mem);
    end
    logic [63:0] a0_0_in;
    logic a0_0_write_en;
    logic a0_0_clk;
    logic a0_0_reset;
    logic [63:0] a0_0_out;
    logic a0_0_done;
    logic [6:0] add0_left;
    logic [6:0] add0_right;
    logic [6:0] add0_out;
    logic [6:0] add1_left;
    logic [6:0] add1_right;
    logic [6:0] add1_out;
    logic [63:0] add2_left;
    logic [63:0] add2_right;
    logic [63:0] add2_out;
    logic [3:0] add5_left;
    logic [3:0] add5_right;
    logic [3:0] add5_out;
    logic [63:0] bin_read0_0_in;
    logic bin_read0_0_write_en;
    logic bin_read0_0_clk;
    logic bin_read0_0_reset;
    logic [63:0] bin_read0_0_out;
    logic bin_read0_0_done;
    logic [63:0] bin_read1_0_in;
    logic bin_read1_0_write_en;
    logic bin_read1_0_clk;
    logic bin_read1_0_reset;
    logic [63:0] bin_read1_0_out;
    logic bin_read1_0_done;
    logic [63:0] bin_read2_0_in;
    logic bin_read2_0_write_en;
    logic bin_read2_0_clk;
    logic bin_read2_0_reset;
    logic [63:0] bin_read2_0_out;
    logic bin_read2_0_done;
    logic [63:0] bin_read3_0_in;
    logic bin_read3_0_write_en;
    logic bin_read3_0_clk;
    logic bin_read3_0_reset;
    logic [63:0] bin_read3_0_out;
    logic bin_read3_0_done;
    logic [63:0] c_img_0_in;
    logic c_img_0_write_en;
    logic c_img_0_clk;
    logic c_img_0_reset;
    logic [63:0] c_img_0_out;
    logic c_img_0_done;
    logic [6:0] c_img_mem0_addr0;
    logic [63:0] c_img_mem0_write_data;
    logic c_img_mem0_write_en;
    logic c_img_mem0_clk;
    logic [63:0] c_img_mem0_read_data;
    logic c_img_mem0_done;
    logic [63:0] c_real_0_in;
    logic c_real_0_write_en;
    logic c_real_0_clk;
    logic c_real_0_reset;
    logic [63:0] c_real_0_out;
    logic c_real_0_done;
    logic [6:0] c_real_mem0_addr0;
    logic [63:0] c_real_mem0_write_data;
    logic c_real_mem0_write_en;
    logic c_real_mem0_clk;
    logic [63:0] c_real_mem0_read_data;
    logic c_real_mem0_done;
    logic [6:0] const0_out;
    logic [6:0] const1_out;
    logic [3:0] const11_out;
    logic [3:0] const12_out;
    logic const13_out;
    logic [3:0] const15_out;
    logic [6:0] const4_out;
    logic [6:0] const5_out;
    logic [6:0] const6_out;
    logic const7_out;
    logic [6:0] const8_out;
    logic [63:0] doub_ixr_0_in;
    logic doub_ixr_0_write_en;
    logic doub_ixr_0_clk;
    logic doub_ixr_0_reset;
    logic [63:0] doub_ixr_0_out;
    logic doub_ixr_0_done;
    logic eq0_left;
    logic eq0_right;
    logic eq0_out;
    logic [63:0] fp_const0_out;
    logic [63:0] fp_const1_out;
    logic [63:0] fp_const2_out;
    logic [63:0] fp_const3_out;
    logic [63:0] gt0_left;
    logic [63:0] gt0_right;
    logic gt0_out;
    logic [6:0] i0_in;
    logic i0_write_en;
    logic i0_clk;
    logic i0_reset;
    logic [6:0] i0_out;
    logic i0_done;
    logic [6:0] i1_in;
    logic i1_write_en;
    logic i1_clk;
    logic i1_reset;
    logic [6:0] i1_out;
    logic i1_done;
    logic [6:0] int_c_img0_addr0;
    logic [63:0] int_c_img0_write_data;
    logic int_c_img0_write_en;
    logic int_c_img0_clk;
    logic [63:0] int_c_img0_read_data;
    logic int_c_img0_done;
    logic [6:0] int_c_real0_addr0;
    logic [63:0] int_c_real0_write_data;
    logic int_c_real0_write_en;
    logic int_c_real0_clk;
    logic [63:0] int_c_real0_read_data;
    logic int_c_real0_done;
    logic [6:0] int_outputs0_addr0;
    logic int_outputs0_write_data;
    logic int_outputs0_write_en;
    logic int_outputs0_clk;
    logic int_outputs0_read_data;
    logic int_outputs0_done;
    logic [6:0] le0_left;
    logic [6:0] le0_right;
    logic le0_out;
    logic [6:0] le1_left;
    logic [6:0] le1_right;
    logic le1_out;
    logic [3:0] le3_left;
    logic [3:0] le3_right;
    logic le3_out;
    logic mult_pipe0_clk;
    logic mult_pipe0_go;
    logic [63:0] mult_pipe0_left;
    logic [63:0] mult_pipe0_right;
    logic [63:0] mult_pipe0_out;
    logic mult_pipe0_done;
    logic mult_pipe1_clk;
    logic mult_pipe1_go;
    logic [63:0] mult_pipe1_left;
    logic [63:0] mult_pipe1_right;
    logic [63:0] mult_pipe1_out;
    logic mult_pipe1_done;
    logic mult_pipe2_clk;
    logic mult_pipe2_go;
    logic [63:0] mult_pipe2_left;
    logic [63:0] mult_pipe2_right;
    logic [63:0] mult_pipe2_out;
    logic mult_pipe2_done;
    logic mult_pipe3_clk;
    logic mult_pipe3_go;
    logic [63:0] mult_pipe3_left;
    logic [63:0] mult_pipe3_right;
    logic [63:0] mult_pipe3_out;
    logic mult_pipe3_done;
    logic [3:0] n_iters0_in;
    logic n_iters0_write_en;
    logic n_iters0_clk;
    logic n_iters0_reset;
    logic [3:0] n_iters0_out;
    logic n_iters0_done;
    logic or0_left;
    logic or0_right;
    logic or0_out;
    logic [6:0] outputs0_addr0;
    logic outputs0_write_data;
    logic outputs0_write_en;
    logic outputs0_clk;
    logic outputs0_read_data;
    logic outputs0_done;
    logic outputs_read0_0_in;
    logic outputs_read0_0_write_en;
    logic outputs_read0_0_clk;
    logic outputs_read0_0_reset;
    logic outputs_read0_0_out;
    logic outputs_read0_0_done;
    logic [6:0] rsh0_left;
    logic [6:0] rsh0_right;
    logic [6:0] rsh0_out;
    logic [63:0] sub0_left;
    logic [63:0] sub0_right;
    logic [63:0] sub0_out;
    logic [63:0] z_img_0_in;
    logic z_img_0_write_en;
    logic z_img_0_clk;
    logic z_img_0_reset;
    logic [63:0] z_img_0_out;
    logic z_img_0_done;
    logic [6:0] z_img_mem0_addr0;
    logic [63:0] z_img_mem0_write_data;
    logic z_img_mem0_write_en;
    logic z_img_mem0_clk;
    logic [63:0] z_img_mem0_read_data;
    logic z_img_mem0_done;
    logic [63:0] z_real_0_in;
    logic z_real_0_write_en;
    logic z_real_0_clk;
    logic z_real_0_reset;
    logic [63:0] z_real_0_out;
    logic z_real_0_done;
    logic [63:0] z_real_2_0_in;
    logic z_real_2_0_write_en;
    logic z_real_2_0_clk;
    logic z_real_2_0_reset;
    logic [63:0] z_real_2_0_out;
    logic z_real_2_0_done;
    logic [6:0] z_real_mem0_addr0;
    logic [63:0] z_real_mem0_write_data;
    logic z_real_mem0_write_en;
    logic z_real_mem0_clk;
    logic [63:0] z_real_mem0_read_data;
    logic z_real_mem0_done;
    logic fsm_in;
    logic fsm_write_en;
    logic fsm_clk;
    logic fsm_reset;
    logic fsm_out;
    logic fsm_done;
    logic incr_left;
    logic incr_right;
    logic incr_out;
    logic [2:0] fsm0_in;
    logic fsm0_write_en;
    logic fsm0_clk;
    logic fsm0_reset;
    logic [2:0] fsm0_out;
    logic fsm0_done;
    logic [2:0] incr0_left;
    logic [2:0] incr0_right;
    logic [2:0] incr0_out;
    logic [2:0] fsm1_in;
    logic fsm1_write_en;
    logic fsm1_clk;
    logic fsm1_reset;
    logic [2:0] fsm1_out;
    logic fsm1_done;
    logic cond_stored_in;
    logic cond_stored_write_en;
    logic cond_stored_clk;
    logic cond_stored_reset;
    logic cond_stored_out;
    logic cond_stored_done;
    logic [2:0] incr1_left;
    logic [2:0] incr1_right;
    logic [2:0] incr1_out;
    logic fsm2_in;
    logic fsm2_write_en;
    logic fsm2_clk;
    logic fsm2_reset;
    logic fsm2_out;
    logic fsm2_done;
    logic incr2_left;
    logic incr2_right;
    logic incr2_out;
    logic [1:0] fsm3_in;
    logic fsm3_write_en;
    logic fsm3_clk;
    logic fsm3_reset;
    logic [1:0] fsm3_out;
    logic fsm3_done;
    logic [1:0] incr3_left;
    logic [1:0] incr3_right;
    logic [1:0] incr3_out;
    logic [1:0] fsm4_in;
    logic fsm4_write_en;
    logic fsm4_clk;
    logic fsm4_reset;
    logic [1:0] fsm4_out;
    logic fsm4_done;
    logic cond_stored0_in;
    logic cond_stored0_write_en;
    logic cond_stored0_clk;
    logic cond_stored0_reset;
    logic cond_stored0_out;
    logic cond_stored0_done;
    logic [1:0] incr4_left;
    logic [1:0] incr4_right;
    logic [1:0] incr4_out;
    logic fsm5_in;
    logic fsm5_write_en;
    logic fsm5_clk;
    logic fsm5_reset;
    logic fsm5_out;
    logic fsm5_done;
    logic incr5_left;
    logic incr5_right;
    logic incr5_out;
    logic fsm6_in;
    logic fsm6_write_en;
    logic fsm6_clk;
    logic fsm6_reset;
    logic fsm6_out;
    logic fsm6_done;
    logic incr6_left;
    logic incr6_right;
    logic incr6_out;
    logic [2:0] fsm7_in;
    logic fsm7_write_en;
    logic fsm7_clk;
    logic fsm7_reset;
    logic [2:0] fsm7_out;
    logic fsm7_done;
    logic [2:0] incr7_left;
    logic [2:0] incr7_right;
    logic [2:0] incr7_out;
    logic [2:0] fsm8_in;
    logic fsm8_write_en;
    logic fsm8_clk;
    logic fsm8_reset;
    logic [2:0] fsm8_out;
    logic fsm8_done;
    logic [2:0] incr8_left;
    logic [2:0] incr8_right;
    logic [2:0] incr8_out;
    logic [2:0] fsm9_in;
    logic fsm9_write_en;
    logic fsm9_clk;
    logic fsm9_reset;
    logic [2:0] fsm9_out;
    logic fsm9_done;
    logic [2:0] incr9_left;
    logic [2:0] incr9_right;
    logic [2:0] incr9_out;
    logic [2:0] fsm10_in;
    logic fsm10_write_en;
    logic fsm10_clk;
    logic fsm10_reset;
    logic [2:0] fsm10_out;
    logic fsm10_done;
    logic [2:0] incr10_left;
    logic [2:0] incr10_right;
    logic [2:0] incr10_out;
    logic [2:0] fsm11_in;
    logic fsm11_write_en;
    logic fsm11_clk;
    logic fsm11_reset;
    logic [2:0] fsm11_out;
    logic fsm11_done;
    logic [2:0] incr11_left;
    logic [2:0] incr11_right;
    logic [2:0] incr11_out;
    logic [3:0] fsm12_in;
    logic fsm12_write_en;
    logic fsm12_clk;
    logic fsm12_reset;
    logic [3:0] fsm12_out;
    logic fsm12_done;
    logic [3:0] incr12_left;
    logic [3:0] incr12_right;
    logic [3:0] incr12_out;
    logic [3:0] fsm13_in;
    logic fsm13_write_en;
    logic fsm13_clk;
    logic fsm13_reset;
    logic [3:0] fsm13_out;
    logic fsm13_done;
    logic cond_stored1_in;
    logic cond_stored1_write_en;
    logic cond_stored1_clk;
    logic cond_stored1_reset;
    logic cond_stored1_out;
    logic cond_stored1_done;
    logic [3:0] incr13_left;
    logic [3:0] incr13_right;
    logic [3:0] incr13_out;
    logic [4:0] fsm14_in;
    logic fsm14_write_en;
    logic fsm14_clk;
    logic fsm14_reset;
    logic [4:0] fsm14_out;
    logic fsm14_done;
    logic [4:0] incr14_left;
    logic [4:0] incr14_right;
    logic [4:0] incr14_out;
    logic [4:0] fsm15_in;
    logic fsm15_write_en;
    logic fsm15_clk;
    logic fsm15_reset;
    logic [4:0] fsm15_out;
    logic fsm15_done;
    logic cond_stored2_in;
    logic cond_stored2_write_en;
    logic cond_stored2_clk;
    logic cond_stored2_reset;
    logic cond_stored2_out;
    logic cond_stored2_done;
    logic [4:0] incr15_left;
    logic [4:0] incr15_right;
    logic [4:0] incr15_out;
    logic [1:0] fsm16_in;
    logic fsm16_write_en;
    logic fsm16_clk;
    logic fsm16_reset;
    logic [1:0] fsm16_out;
    logic fsm16_done;
    logic [1:0] incr16_left;
    logic [1:0] incr16_right;
    logic [1:0] incr16_out;
    logic [2:0] fsm17_in;
    logic fsm17_write_en;
    logic fsm17_clk;
    logic fsm17_reset;
    logic [2:0] fsm17_out;
    logic fsm17_done;
    logic cond_stored3_in;
    logic cond_stored3_write_en;
    logic cond_stored3_clk;
    logic cond_stored3_reset;
    logic cond_stored3_out;
    logic cond_stored3_done;
    logic [2:0] incr17_left;
    logic [2:0] incr17_right;
    logic [2:0] incr17_out;
    logic [1:0] fsm18_in;
    logic fsm18_write_en;
    logic fsm18_clk;
    logic fsm18_reset;
    logic [1:0] fsm18_out;
    logic fsm18_done;
    logic pd_in;
    logic pd_write_en;
    logic pd_clk;
    logic pd_reset;
    logic pd_out;
    logic pd_done;
    logic [1:0] fsm19_in;
    logic fsm19_write_en;
    logic fsm19_clk;
    logic fsm19_reset;
    logic [1:0] fsm19_out;
    logic fsm19_done;
    logic pd0_in;
    logic pd0_write_en;
    logic pd0_clk;
    logic pd0_reset;
    logic pd0_out;
    logic pd0_done;
    logic cs_wh_in;
    logic cs_wh_write_en;
    logic cs_wh_clk;
    logic cs_wh_reset;
    logic cs_wh_out;
    logic cs_wh_done;
    logic [3:0] fsm20_in;
    logic fsm20_write_en;
    logic fsm20_clk;
    logic fsm20_reset;
    logic [3:0] fsm20_out;
    logic fsm20_done;
    initial begin
        a0_0_in = 64'd0;
        a0_0_write_en = 1'd0;
        a0_0_clk = 1'd0;
        a0_0_reset = 1'd0;
        add0_left = 7'd0;
        add0_right = 7'd0;
        add1_left = 7'd0;
        add1_right = 7'd0;
        add2_left = 64'd0;
        add2_right = 64'd0;
        add5_left = 4'd0;
        add5_right = 4'd0;
        bin_read0_0_in = 64'd0;
        bin_read0_0_write_en = 1'd0;
        bin_read0_0_clk = 1'd0;
        bin_read0_0_reset = 1'd0;
        bin_read1_0_in = 64'd0;
        bin_read1_0_write_en = 1'd0;
        bin_read1_0_clk = 1'd0;
        bin_read1_0_reset = 1'd0;
        bin_read2_0_in = 64'd0;
        bin_read2_0_write_en = 1'd0;
        bin_read2_0_clk = 1'd0;
        bin_read2_0_reset = 1'd0;
        bin_read3_0_in = 64'd0;
        bin_read3_0_write_en = 1'd0;
        bin_read3_0_clk = 1'd0;
        bin_read3_0_reset = 1'd0;
        c_img_0_in = 64'd0;
        c_img_0_write_en = 1'd0;
        c_img_0_clk = 1'd0;
        c_img_0_reset = 1'd0;
        c_img_mem0_addr0 = 7'd0;
        c_img_mem0_write_data = 64'd0;
        c_img_mem0_write_en = 1'd0;
        c_img_mem0_clk = 1'd0;
        c_real_0_in = 64'd0;
        c_real_0_write_en = 1'd0;
        c_real_0_clk = 1'd0;
        c_real_0_reset = 1'd0;
        c_real_mem0_addr0 = 7'd0;
        c_real_mem0_write_data = 64'd0;
        c_real_mem0_write_en = 1'd0;
        c_real_mem0_clk = 1'd0;
        doub_ixr_0_in = 64'd0;
        doub_ixr_0_write_en = 1'd0;
        doub_ixr_0_clk = 1'd0;
        doub_ixr_0_reset = 1'd0;
        eq0_left = 1'd0;
        eq0_right = 1'd0;
        gt0_left = 64'd0;
        gt0_right = 64'd0;
        i0_in = 7'd0;
        i0_write_en = 1'd0;
        i0_clk = 1'd0;
        i0_reset = 1'd0;
        i1_in = 7'd0;
        i1_write_en = 1'd0;
        i1_clk = 1'd0;
        i1_reset = 1'd0;
        int_c_img0_addr0 = 7'd0;
        int_c_img0_write_data = 64'd0;
        int_c_img0_write_en = 1'd0;
        int_c_img0_clk = 1'd0;
        int_c_real0_addr0 = 7'd0;
        int_c_real0_write_data = 64'd0;
        int_c_real0_write_en = 1'd0;
        int_c_real0_clk = 1'd0;
        int_outputs0_addr0 = 7'd0;
        int_outputs0_write_data = 1'd0;
        int_outputs0_write_en = 1'd0;
        int_outputs0_clk = 1'd0;
        le0_left = 7'd0;
        le0_right = 7'd0;
        le1_left = 7'd0;
        le1_right = 7'd0;
        le3_left = 4'd0;
        le3_right = 4'd0;
        mult_pipe0_clk = 1'd0;
        mult_pipe0_go = 1'd0;
        mult_pipe0_left = 64'd0;
        mult_pipe0_right = 64'd0;
        mult_pipe1_clk = 1'd0;
        mult_pipe1_go = 1'd0;
        mult_pipe1_left = 64'd0;
        mult_pipe1_right = 64'd0;
        mult_pipe2_clk = 1'd0;
        mult_pipe2_go = 1'd0;
        mult_pipe2_left = 64'd0;
        mult_pipe2_right = 64'd0;
        mult_pipe3_clk = 1'd0;
        mult_pipe3_go = 1'd0;
        mult_pipe3_left = 64'd0;
        mult_pipe3_right = 64'd0;
        n_iters0_in = 4'd0;
        n_iters0_write_en = 1'd0;
        n_iters0_clk = 1'd0;
        n_iters0_reset = 1'd0;
        or0_left = 1'd0;
        or0_right = 1'd0;
        outputs0_addr0 = 7'd0;
        outputs0_write_data = 1'd0;
        outputs0_write_en = 1'd0;
        outputs0_clk = 1'd0;
        outputs_read0_0_in = 1'd0;
        outputs_read0_0_write_en = 1'd0;
        outputs_read0_0_clk = 1'd0;
        outputs_read0_0_reset = 1'd0;
        rsh0_left = 7'd0;
        rsh0_right = 7'd0;
        sub0_left = 64'd0;
        sub0_right = 64'd0;
        z_img_0_in = 64'd0;
        z_img_0_write_en = 1'd0;
        z_img_0_clk = 1'd0;
        z_img_0_reset = 1'd0;
        z_img_mem0_addr0 = 7'd0;
        z_img_mem0_write_data = 64'd0;
        z_img_mem0_write_en = 1'd0;
        z_img_mem0_clk = 1'd0;
        z_real_0_in = 64'd0;
        z_real_0_write_en = 1'd0;
        z_real_0_clk = 1'd0;
        z_real_0_reset = 1'd0;
        z_real_2_0_in = 64'd0;
        z_real_2_0_write_en = 1'd0;
        z_real_2_0_clk = 1'd0;
        z_real_2_0_reset = 1'd0;
        z_real_mem0_addr0 = 7'd0;
        z_real_mem0_write_data = 64'd0;
        z_real_mem0_write_en = 1'd0;
        z_real_mem0_clk = 1'd0;
        fsm_in = 1'd0;
        fsm_write_en = 1'd0;
        fsm_clk = 1'd0;
        fsm_reset = 1'd0;
        incr_left = 1'd0;
        incr_right = 1'd0;
        fsm0_in = 3'd0;
        fsm0_write_en = 1'd0;
        fsm0_clk = 1'd0;
        fsm0_reset = 1'd0;
        incr0_left = 3'd0;
        incr0_right = 3'd0;
        fsm1_in = 3'd0;
        fsm1_write_en = 1'd0;
        fsm1_clk = 1'd0;
        fsm1_reset = 1'd0;
        cond_stored_in = 1'd0;
        cond_stored_write_en = 1'd0;
        cond_stored_clk = 1'd0;
        cond_stored_reset = 1'd0;
        incr1_left = 3'd0;
        incr1_right = 3'd0;
        fsm2_in = 1'd0;
        fsm2_write_en = 1'd0;
        fsm2_clk = 1'd0;
        fsm2_reset = 1'd0;
        incr2_left = 1'd0;
        incr2_right = 1'd0;
        fsm3_in = 2'd0;
        fsm3_write_en = 1'd0;
        fsm3_clk = 1'd0;
        fsm3_reset = 1'd0;
        incr3_left = 2'd0;
        incr3_right = 2'd0;
        fsm4_in = 2'd0;
        fsm4_write_en = 1'd0;
        fsm4_clk = 1'd0;
        fsm4_reset = 1'd0;
        cond_stored0_in = 1'd0;
        cond_stored0_write_en = 1'd0;
        cond_stored0_clk = 1'd0;
        cond_stored0_reset = 1'd0;
        incr4_left = 2'd0;
        incr4_right = 2'd0;
        fsm5_in = 1'd0;
        fsm5_write_en = 1'd0;
        fsm5_clk = 1'd0;
        fsm5_reset = 1'd0;
        incr5_left = 1'd0;
        incr5_right = 1'd0;
        fsm6_in = 1'd0;
        fsm6_write_en = 1'd0;
        fsm6_clk = 1'd0;
        fsm6_reset = 1'd0;
        incr6_left = 1'd0;
        incr6_right = 1'd0;
        fsm7_in = 3'd0;
        fsm7_write_en = 1'd0;
        fsm7_clk = 1'd0;
        fsm7_reset = 1'd0;
        incr7_left = 3'd0;
        incr7_right = 3'd0;
        fsm8_in = 3'd0;
        fsm8_write_en = 1'd0;
        fsm8_clk = 1'd0;
        fsm8_reset = 1'd0;
        incr8_left = 3'd0;
        incr8_right = 3'd0;
        fsm9_in = 3'd0;
        fsm9_write_en = 1'd0;
        fsm9_clk = 1'd0;
        fsm9_reset = 1'd0;
        incr9_left = 3'd0;
        incr9_right = 3'd0;
        fsm10_in = 3'd0;
        fsm10_write_en = 1'd0;
        fsm10_clk = 1'd0;
        fsm10_reset = 1'd0;
        incr10_left = 3'd0;
        incr10_right = 3'd0;
        fsm11_in = 3'd0;
        fsm11_write_en = 1'd0;
        fsm11_clk = 1'd0;
        fsm11_reset = 1'd0;
        incr11_left = 3'd0;
        incr11_right = 3'd0;
        fsm12_in = 4'd0;
        fsm12_write_en = 1'd0;
        fsm12_clk = 1'd0;
        fsm12_reset = 1'd0;
        incr12_left = 4'd0;
        incr12_right = 4'd0;
        fsm13_in = 4'd0;
        fsm13_write_en = 1'd0;
        fsm13_clk = 1'd0;
        fsm13_reset = 1'd0;
        cond_stored1_in = 1'd0;
        cond_stored1_write_en = 1'd0;
        cond_stored1_clk = 1'd0;
        cond_stored1_reset = 1'd0;
        incr13_left = 4'd0;
        incr13_right = 4'd0;
        fsm14_in = 5'd0;
        fsm14_write_en = 1'd0;
        fsm14_clk = 1'd0;
        fsm14_reset = 1'd0;
        incr14_left = 5'd0;
        incr14_right = 5'd0;
        fsm15_in = 5'd0;
        fsm15_write_en = 1'd0;
        fsm15_clk = 1'd0;
        fsm15_reset = 1'd0;
        cond_stored2_in = 1'd0;
        cond_stored2_write_en = 1'd0;
        cond_stored2_clk = 1'd0;
        cond_stored2_reset = 1'd0;
        incr15_left = 5'd0;
        incr15_right = 5'd0;
        fsm16_in = 2'd0;
        fsm16_write_en = 1'd0;
        fsm16_clk = 1'd0;
        fsm16_reset = 1'd0;
        incr16_left = 2'd0;
        incr16_right = 2'd0;
        fsm17_in = 3'd0;
        fsm17_write_en = 1'd0;
        fsm17_clk = 1'd0;
        fsm17_reset = 1'd0;
        cond_stored3_in = 1'd0;
        cond_stored3_write_en = 1'd0;
        cond_stored3_clk = 1'd0;
        cond_stored3_reset = 1'd0;
        incr17_left = 3'd0;
        incr17_right = 3'd0;
        fsm18_in = 2'd0;
        fsm18_write_en = 1'd0;
        fsm18_clk = 1'd0;
        fsm18_reset = 1'd0;
        pd_in = 1'd0;
        pd_write_en = 1'd0;
        pd_clk = 1'd0;
        pd_reset = 1'd0;
        fsm19_in = 2'd0;
        fsm19_write_en = 1'd0;
        fsm19_clk = 1'd0;
        fsm19_reset = 1'd0;
        pd0_in = 1'd0;
        pd0_write_en = 1'd0;
        pd0_clk = 1'd0;
        pd0_reset = 1'd0;
        cs_wh_in = 1'd0;
        cs_wh_write_en = 1'd0;
        cs_wh_clk = 1'd0;
        cs_wh_reset = 1'd0;
        fsm20_in = 4'd0;
        fsm20_write_en = 1'd0;
        fsm20_clk = 1'd0;
        fsm20_reset = 1'd0;
    end
    std_reg # (
        .WIDTH(64)
    ) a0_0 (
        .clk(a0_0_clk),
        .done(a0_0_done),
        .in(a0_0_in),
        .out(a0_0_out),
        .reset(a0_0_reset),
        .write_en(a0_0_write_en)
    );
    std_add # (
        .WIDTH(7)
    ) add0 (
        .left(add0_left),
        .out(add0_out),
        .right(add0_right)
    );
    std_add # (
        .WIDTH(7)
    ) add1 (
        .left(add1_left),
        .out(add1_out),
        .right(add1_right)
    );
    std_fp_sadd # (
        .FRAC_WIDTH(32),
        .INT_WIDTH(32),
        .WIDTH(64)
    ) add2 (
        .left(add2_left),
        .out(add2_out),
        .right(add2_right)
    );
    std_add # (
        .WIDTH(4)
    ) add5 (
        .left(add5_left),
        .out(add5_out),
        .right(add5_right)
    );
    std_reg # (
        .WIDTH(64)
    ) bin_read0_0 (
        .clk(bin_read0_0_clk),
        .done(bin_read0_0_done),
        .in(bin_read0_0_in),
        .out(bin_read0_0_out),
        .reset(bin_read0_0_reset),
        .write_en(bin_read0_0_write_en)
    );
    std_reg # (
        .WIDTH(64)
    ) bin_read1_0 (
        .clk(bin_read1_0_clk),
        .done(bin_read1_0_done),
        .in(bin_read1_0_in),
        .out(bin_read1_0_out),
        .reset(bin_read1_0_reset),
        .write_en(bin_read1_0_write_en)
    );
    std_reg # (
        .WIDTH(64)
    ) bin_read2_0 (
        .clk(bin_read2_0_clk),
        .done(bin_read2_0_done),
        .in(bin_read2_0_in),
        .out(bin_read2_0_out),
        .reset(bin_read2_0_reset),
        .write_en(bin_read2_0_write_en)
    );
    std_reg # (
        .WIDTH(64)
    ) bin_read3_0 (
        .clk(bin_read3_0_clk),
        .done(bin_read3_0_done),
        .in(bin_read3_0_in),
        .out(bin_read3_0_out),
        .reset(bin_read3_0_reset),
        .write_en(bin_read3_0_write_en)
    );
    std_reg # (
        .WIDTH(64)
    ) c_img_0 (
        .clk(c_img_0_clk),
        .done(c_img_0_done),
        .in(c_img_0_in),
        .out(c_img_0_out),
        .reset(c_img_0_reset),
        .write_en(c_img_0_write_en)
    );
    std_mem_d1 # (
        .IDX_SIZE(7),
        .SIZE(64),
        .WIDTH(64)
    ) c_img_mem0 (
        .addr0(c_img_mem0_addr0),
        .clk(c_img_mem0_clk),
        .done(c_img_mem0_done),
        .read_data(c_img_mem0_read_data),
        .write_data(c_img_mem0_write_data),
        .write_en(c_img_mem0_write_en)
    );
    std_reg # (
        .WIDTH(64)
    ) c_real_0 (
        .clk(c_real_0_clk),
        .done(c_real_0_done),
        .in(c_real_0_in),
        .out(c_real_0_out),
        .reset(c_real_0_reset),
        .write_en(c_real_0_write_en)
    );
    std_mem_d1 # (
        .IDX_SIZE(7),
        .SIZE(64),
        .WIDTH(64)
    ) c_real_mem0 (
        .addr0(c_real_mem0_addr0),
        .clk(c_real_mem0_clk),
        .done(c_real_mem0_done),
        .read_data(c_real_mem0_read_data),
        .write_data(c_real_mem0_write_data),
        .write_en(c_real_mem0_write_en)
    );
    std_const # (
        .VALUE(0),
        .WIDTH(7)
    ) const0 (
        .out(const0_out)
    );
    std_const # (
        .VALUE(63),
        .WIDTH(7)
    ) const1 (
        .out(const1_out)
    );
    std_const # (
        .VALUE(0),
        .WIDTH(4)
    ) const11 (
        .out(const11_out)
    );
    std_const # (
        .VALUE(7),
        .WIDTH(4)
    ) const12 (
        .out(const12_out)
    );
    std_const # (
        .VALUE(0),
        .WIDTH(1)
    ) const13 (
        .out(const13_out)
    );
    std_const # (
        .VALUE(1),
        .WIDTH(4)
    ) const15 (
        .out(const15_out)
    );
    std_const # (
        .VALUE(1),
        .WIDTH(7)
    ) const4 (
        .out(const4_out)
    );
    std_const # (
        .VALUE(0),
        .WIDTH(7)
    ) const5 (
        .out(const5_out)
    );
    std_const # (
        .VALUE(63),
        .WIDTH(7)
    ) const6 (
        .out(const6_out)
    );
    std_const # (
        .VALUE(1),
        .WIDTH(1)
    ) const7 (
        .out(const7_out)
    );
    std_const # (
        .VALUE(1),
        .WIDTH(7)
    ) const8 (
        .out(const8_out)
    );
    std_reg # (
        .WIDTH(64)
    ) doub_ixr_0 (
        .clk(doub_ixr_0_clk),
        .done(doub_ixr_0_done),
        .in(doub_ixr_0_in),
        .out(doub_ixr_0_out),
        .reset(doub_ixr_0_reset),
        .write_en(doub_ixr_0_write_en)
    );
    std_seq # (
        .WIDTH(1)
    ) eq0 (
        .left(eq0_left),
        .out(eq0_out),
        .right(eq0_right)
    );
    std_const # (
        .VALUE(0),
        .WIDTH(64)
    ) fp_const0 (
        .out(fp_const0_out)
    );
    std_const # (
        .VALUE(0),
        .WIDTH(64)
    ) fp_const1 (
        .out(fp_const1_out)
    );
    std_const # (
        .VALUE(17179869184),
        .WIDTH(64)
    ) fp_const2 (
        .out(fp_const2_out)
    );
    std_const # (
        .VALUE(8589934592),
        .WIDTH(64)
    ) fp_const3 (
        .out(fp_const3_out)
    );
    std_fp_sgt # (
        .FRAC_WIDTH(32),
        .INT_WIDTH(32),
        .WIDTH(64)
    ) gt0 (
        .left(gt0_left),
        .out(gt0_out),
        .right(gt0_right)
    );
    std_reg # (
        .WIDTH(7)
    ) i0 (
        .clk(i0_clk),
        .done(i0_done),
        .in(i0_in),
        .out(i0_out),
        .reset(i0_reset),
        .write_en(i0_write_en)
    );
    std_reg # (
        .WIDTH(7)
    ) i1 (
        .clk(i1_clk),
        .done(i1_done),
        .in(i1_in),
        .out(i1_out),
        .reset(i1_reset),
        .write_en(i1_write_en)
    );
    std_mem_d1 # (
        .IDX_SIZE(7),
        .SIZE(64),
        .WIDTH(64)
    ) int_c_img0 (
        .addr0(int_c_img0_addr0),
        .clk(int_c_img0_clk),
        .done(int_c_img0_done),
        .read_data(int_c_img0_read_data),
        .write_data(int_c_img0_write_data),
        .write_en(int_c_img0_write_en)
    );
    std_mem_d1 # (
        .IDX_SIZE(7),
        .SIZE(64),
        .WIDTH(64)
    ) int_c_real0 (
        .addr0(int_c_real0_addr0),
        .clk(int_c_real0_clk),
        .done(int_c_real0_done),
        .read_data(int_c_real0_read_data),
        .write_data(int_c_real0_write_data),
        .write_en(int_c_real0_write_en)
    );
    std_mem_d1 # (
        .IDX_SIZE(7),
        .SIZE(64),
        .WIDTH(1)
    ) int_outputs0 (
        .addr0(int_outputs0_addr0),
        .clk(int_outputs0_clk),
        .done(int_outputs0_done),
        .read_data(int_outputs0_read_data),
        .write_data(int_outputs0_write_data),
        .write_en(int_outputs0_write_en)
    );
    std_le # (
        .WIDTH(7)
    ) le0 (
        .left(le0_left),
        .out(le0_out),
        .right(le0_right)
    );
    std_le # (
        .WIDTH(7)
    ) le1 (
        .left(le1_left),
        .out(le1_out),
        .right(le1_right)
    );
    std_le # (
        .WIDTH(4)
    ) le3 (
        .left(le3_left),
        .out(le3_out),
        .right(le3_right)
    );
    std_fp_smult_pipe # (
        .FRAC_WIDTH(32),
        .INT_WIDTH(32),
        .WIDTH(64)
    ) mult_pipe0 (
        .clk(mult_pipe0_clk),
        .done(mult_pipe0_done),
        .go(mult_pipe0_go),
        .left(mult_pipe0_left),
        .out(mult_pipe0_out),
        .right(mult_pipe0_right)
    );
    std_fp_smult_pipe # (
        .FRAC_WIDTH(32),
        .INT_WIDTH(32),
        .WIDTH(64)
    ) mult_pipe1 (
        .clk(mult_pipe1_clk),
        .done(mult_pipe1_done),
        .go(mult_pipe1_go),
        .left(mult_pipe1_left),
        .out(mult_pipe1_out),
        .right(mult_pipe1_right)
    );
    std_fp_smult_pipe # (
        .FRAC_WIDTH(32),
        .INT_WIDTH(32),
        .WIDTH(64)
    ) mult_pipe2 (
        .clk(mult_pipe2_clk),
        .done(mult_pipe2_done),
        .go(mult_pipe2_go),
        .left(mult_pipe2_left),
        .out(mult_pipe2_out),
        .right(mult_pipe2_right)
    );
    std_fp_smult_pipe # (
        .FRAC_WIDTH(32),
        .INT_WIDTH(32),
        .WIDTH(64)
    ) mult_pipe3 (
        .clk(mult_pipe3_clk),
        .done(mult_pipe3_done),
        .go(mult_pipe3_go),
        .left(mult_pipe3_left),
        .out(mult_pipe3_out),
        .right(mult_pipe3_right)
    );
    std_reg # (
        .WIDTH(4)
    ) n_iters0 (
        .clk(n_iters0_clk),
        .done(n_iters0_done),
        .in(n_iters0_in),
        .out(n_iters0_out),
        .reset(n_iters0_reset),
        .write_en(n_iters0_write_en)
    );
    std_or # (
        .WIDTH(1)
    ) or0 (
        .left(or0_left),
        .out(or0_out),
        .right(or0_right)
    );
    std_mem_d1 # (
        .IDX_SIZE(7),
        .SIZE(64),
        .WIDTH(1)
    ) outputs0 (
        .addr0(outputs0_addr0),
        .clk(outputs0_clk),
        .done(outputs0_done),
        .read_data(outputs0_read_data),
        .write_data(outputs0_write_data),
        .write_en(outputs0_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) outputs_read0_0 (
        .clk(outputs_read0_0_clk),
        .done(outputs_read0_0_done),
        .in(outputs_read0_0_in),
        .out(outputs_read0_0_out),
        .reset(outputs_read0_0_reset),
        .write_en(outputs_read0_0_write_en)
    );
    std_rsh # (
        .WIDTH(7)
    ) rsh0 (
        .left(rsh0_left),
        .out(rsh0_out),
        .right(rsh0_right)
    );
    std_fp_ssub # (
        .FRAC_WIDTH(32),
        .INT_WIDTH(32),
        .WIDTH(64)
    ) sub0 (
        .left(sub0_left),
        .out(sub0_out),
        .right(sub0_right)
    );
    std_reg # (
        .WIDTH(64)
    ) z_img_0 (
        .clk(z_img_0_clk),
        .done(z_img_0_done),
        .in(z_img_0_in),
        .out(z_img_0_out),
        .reset(z_img_0_reset),
        .write_en(z_img_0_write_en)
    );
    std_mem_d1 # (
        .IDX_SIZE(7),
        .SIZE(64),
        .WIDTH(64)
    ) z_img_mem0 (
        .addr0(z_img_mem0_addr0),
        .clk(z_img_mem0_clk),
        .done(z_img_mem0_done),
        .read_data(z_img_mem0_read_data),
        .write_data(z_img_mem0_write_data),
        .write_en(z_img_mem0_write_en)
    );
    std_reg # (
        .WIDTH(64)
    ) z_real_0 (
        .clk(z_real_0_clk),
        .done(z_real_0_done),
        .in(z_real_0_in),
        .out(z_real_0_out),
        .reset(z_real_0_reset),
        .write_en(z_real_0_write_en)
    );
    std_reg # (
        .WIDTH(64)
    ) z_real_2_0 (
        .clk(z_real_2_0_clk),
        .done(z_real_2_0_done),
        .in(z_real_2_0_in),
        .out(z_real_2_0_out),
        .reset(z_real_2_0_reset),
        .write_en(z_real_2_0_write_en)
    );
    std_mem_d1 # (
        .IDX_SIZE(7),
        .SIZE(64),
        .WIDTH(64)
    ) z_real_mem0 (
        .addr0(z_real_mem0_addr0),
        .clk(z_real_mem0_clk),
        .done(z_real_mem0_done),
        .read_data(z_real_mem0_read_data),
        .write_data(z_real_mem0_write_data),
        .write_en(z_real_mem0_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) fsm (
        .clk(fsm_clk),
        .done(fsm_done),
        .in(fsm_in),
        .out(fsm_out),
        .reset(fsm_reset),
        .write_en(fsm_write_en)
    );
    std_add # (
        .WIDTH(1)
    ) incr (
        .left(incr_left),
        .out(incr_out),
        .right(incr_right)
    );
    std_reg # (
        .WIDTH(3)
    ) fsm0 (
        .clk(fsm0_clk),
        .done(fsm0_done),
        .in(fsm0_in),
        .out(fsm0_out),
        .reset(fsm0_reset),
        .write_en(fsm0_write_en)
    );
    std_add # (
        .WIDTH(3)
    ) incr0 (
        .left(incr0_left),
        .out(incr0_out),
        .right(incr0_right)
    );
    std_reg # (
        .WIDTH(3)
    ) fsm1 (
        .clk(fsm1_clk),
        .done(fsm1_done),
        .in(fsm1_in),
        .out(fsm1_out),
        .reset(fsm1_reset),
        .write_en(fsm1_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) cond_stored (
        .clk(cond_stored_clk),
        .done(cond_stored_done),
        .in(cond_stored_in),
        .out(cond_stored_out),
        .reset(cond_stored_reset),
        .write_en(cond_stored_write_en)
    );
    std_add # (
        .WIDTH(3)
    ) incr1 (
        .left(incr1_left),
        .out(incr1_out),
        .right(incr1_right)
    );
    std_reg # (
        .WIDTH(1)
    ) fsm2 (
        .clk(fsm2_clk),
        .done(fsm2_done),
        .in(fsm2_in),
        .out(fsm2_out),
        .reset(fsm2_reset),
        .write_en(fsm2_write_en)
    );
    std_add # (
        .WIDTH(1)
    ) incr2 (
        .left(incr2_left),
        .out(incr2_out),
        .right(incr2_right)
    );
    std_reg # (
        .WIDTH(2)
    ) fsm3 (
        .clk(fsm3_clk),
        .done(fsm3_done),
        .in(fsm3_in),
        .out(fsm3_out),
        .reset(fsm3_reset),
        .write_en(fsm3_write_en)
    );
    std_add # (
        .WIDTH(2)
    ) incr3 (
        .left(incr3_left),
        .out(incr3_out),
        .right(incr3_right)
    );
    std_reg # (
        .WIDTH(2)
    ) fsm4 (
        .clk(fsm4_clk),
        .done(fsm4_done),
        .in(fsm4_in),
        .out(fsm4_out),
        .reset(fsm4_reset),
        .write_en(fsm4_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) cond_stored0 (
        .clk(cond_stored0_clk),
        .done(cond_stored0_done),
        .in(cond_stored0_in),
        .out(cond_stored0_out),
        .reset(cond_stored0_reset),
        .write_en(cond_stored0_write_en)
    );
    std_add # (
        .WIDTH(2)
    ) incr4 (
        .left(incr4_left),
        .out(incr4_out),
        .right(incr4_right)
    );
    std_reg # (
        .WIDTH(1)
    ) fsm5 (
        .clk(fsm5_clk),
        .done(fsm5_done),
        .in(fsm5_in),
        .out(fsm5_out),
        .reset(fsm5_reset),
        .write_en(fsm5_write_en)
    );
    std_add # (
        .WIDTH(1)
    ) incr5 (
        .left(incr5_left),
        .out(incr5_out),
        .right(incr5_right)
    );
    std_reg # (
        .WIDTH(1)
    ) fsm6 (
        .clk(fsm6_clk),
        .done(fsm6_done),
        .in(fsm6_in),
        .out(fsm6_out),
        .reset(fsm6_reset),
        .write_en(fsm6_write_en)
    );
    std_add # (
        .WIDTH(1)
    ) incr6 (
        .left(incr6_left),
        .out(incr6_out),
        .right(incr6_right)
    );
    std_reg # (
        .WIDTH(3)
    ) fsm7 (
        .clk(fsm7_clk),
        .done(fsm7_done),
        .in(fsm7_in),
        .out(fsm7_out),
        .reset(fsm7_reset),
        .write_en(fsm7_write_en)
    );
    std_add # (
        .WIDTH(3)
    ) incr7 (
        .left(incr7_left),
        .out(incr7_out),
        .right(incr7_right)
    );
    std_reg # (
        .WIDTH(3)
    ) fsm8 (
        .clk(fsm8_clk),
        .done(fsm8_done),
        .in(fsm8_in),
        .out(fsm8_out),
        .reset(fsm8_reset),
        .write_en(fsm8_write_en)
    );
    std_add # (
        .WIDTH(3)
    ) incr8 (
        .left(incr8_left),
        .out(incr8_out),
        .right(incr8_right)
    );
    std_reg # (
        .WIDTH(3)
    ) fsm9 (
        .clk(fsm9_clk),
        .done(fsm9_done),
        .in(fsm9_in),
        .out(fsm9_out),
        .reset(fsm9_reset),
        .write_en(fsm9_write_en)
    );
    std_add # (
        .WIDTH(3)
    ) incr9 (
        .left(incr9_left),
        .out(incr9_out),
        .right(incr9_right)
    );
    std_reg # (
        .WIDTH(3)
    ) fsm10 (
        .clk(fsm10_clk),
        .done(fsm10_done),
        .in(fsm10_in),
        .out(fsm10_out),
        .reset(fsm10_reset),
        .write_en(fsm10_write_en)
    );
    std_add # (
        .WIDTH(3)
    ) incr10 (
        .left(incr10_left),
        .out(incr10_out),
        .right(incr10_right)
    );
    std_reg # (
        .WIDTH(3)
    ) fsm11 (
        .clk(fsm11_clk),
        .done(fsm11_done),
        .in(fsm11_in),
        .out(fsm11_out),
        .reset(fsm11_reset),
        .write_en(fsm11_write_en)
    );
    std_add # (
        .WIDTH(3)
    ) incr11 (
        .left(incr11_left),
        .out(incr11_out),
        .right(incr11_right)
    );
    std_reg # (
        .WIDTH(4)
    ) fsm12 (
        .clk(fsm12_clk),
        .done(fsm12_done),
        .in(fsm12_in),
        .out(fsm12_out),
        .reset(fsm12_reset),
        .write_en(fsm12_write_en)
    );
    std_add # (
        .WIDTH(4)
    ) incr12 (
        .left(incr12_left),
        .out(incr12_out),
        .right(incr12_right)
    );
    std_reg # (
        .WIDTH(4)
    ) fsm13 (
        .clk(fsm13_clk),
        .done(fsm13_done),
        .in(fsm13_in),
        .out(fsm13_out),
        .reset(fsm13_reset),
        .write_en(fsm13_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) cond_stored1 (
        .clk(cond_stored1_clk),
        .done(cond_stored1_done),
        .in(cond_stored1_in),
        .out(cond_stored1_out),
        .reset(cond_stored1_reset),
        .write_en(cond_stored1_write_en)
    );
    std_add # (
        .WIDTH(4)
    ) incr13 (
        .left(incr13_left),
        .out(incr13_out),
        .right(incr13_right)
    );
    std_reg # (
        .WIDTH(5)
    ) fsm14 (
        .clk(fsm14_clk),
        .done(fsm14_done),
        .in(fsm14_in),
        .out(fsm14_out),
        .reset(fsm14_reset),
        .write_en(fsm14_write_en)
    );
    std_add # (
        .WIDTH(5)
    ) incr14 (
        .left(incr14_left),
        .out(incr14_out),
        .right(incr14_right)
    );
    std_reg # (
        .WIDTH(5)
    ) fsm15 (
        .clk(fsm15_clk),
        .done(fsm15_done),
        .in(fsm15_in),
        .out(fsm15_out),
        .reset(fsm15_reset),
        .write_en(fsm15_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) cond_stored2 (
        .clk(cond_stored2_clk),
        .done(cond_stored2_done),
        .in(cond_stored2_in),
        .out(cond_stored2_out),
        .reset(cond_stored2_reset),
        .write_en(cond_stored2_write_en)
    );
    std_add # (
        .WIDTH(5)
    ) incr15 (
        .left(incr15_left),
        .out(incr15_out),
        .right(incr15_right)
    );
    std_reg # (
        .WIDTH(2)
    ) fsm16 (
        .clk(fsm16_clk),
        .done(fsm16_done),
        .in(fsm16_in),
        .out(fsm16_out),
        .reset(fsm16_reset),
        .write_en(fsm16_write_en)
    );
    std_add # (
        .WIDTH(2)
    ) incr16 (
        .left(incr16_left),
        .out(incr16_out),
        .right(incr16_right)
    );
    std_reg # (
        .WIDTH(3)
    ) fsm17 (
        .clk(fsm17_clk),
        .done(fsm17_done),
        .in(fsm17_in),
        .out(fsm17_out),
        .reset(fsm17_reset),
        .write_en(fsm17_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) cond_stored3 (
        .clk(cond_stored3_clk),
        .done(cond_stored3_done),
        .in(cond_stored3_in),
        .out(cond_stored3_out),
        .reset(cond_stored3_reset),
        .write_en(cond_stored3_write_en)
    );
    std_add # (
        .WIDTH(3)
    ) incr17 (
        .left(incr17_left),
        .out(incr17_out),
        .right(incr17_right)
    );
    std_reg # (
        .WIDTH(2)
    ) fsm18 (
        .clk(fsm18_clk),
        .done(fsm18_done),
        .in(fsm18_in),
        .out(fsm18_out),
        .reset(fsm18_reset),
        .write_en(fsm18_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) pd (
        .clk(pd_clk),
        .done(pd_done),
        .in(pd_in),
        .out(pd_out),
        .reset(pd_reset),
        .write_en(pd_write_en)
    );
    std_reg # (
        .WIDTH(2)
    ) fsm19 (
        .clk(fsm19_clk),
        .done(fsm19_done),
        .in(fsm19_in),
        .out(fsm19_out),
        .reset(fsm19_reset),
        .write_en(fsm19_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) pd0 (
        .clk(pd0_clk),
        .done(pd0_done),
        .in(pd0_in),
        .out(pd0_out),
        .reset(pd0_reset),
        .write_en(pd0_write_en)
    );
    std_reg # (
        .WIDTH(1)
    ) cs_wh (
        .clk(cs_wh_clk),
        .done(cs_wh_done),
        .in(cs_wh_in),
        .out(cs_wh_out),
        .reset(cs_wh_reset),
        .write_en(cs_wh_write_en)
    );
    std_reg # (
        .WIDTH(4)
    ) fsm20 (
        .clk(fsm20_clk),
        .done(fsm20_done),
        .in(fsm20_in),
        .out(fsm20_out),
        .reset(fsm20_reset),
        .write_en(fsm20_write_en)
    );
    assign done =
     fsm20_out == 4'd10 ? 1'd1 : 1'd0;
    assign a0_0_clk =
     1'b1 ? clk : 1'd0;
    assign a0_0_in =
     fsm12_out == 4'd12 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm12_out == 4'd1 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? add2_out :
     fsm7_out == 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? bin_read0_0_out :
     fsm_out < 1'd1 & fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? int_c_img0_read_data : 64'd0;
    assign a0_0_write_en =
     fsm12_out == 4'd12 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm7_out == 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm12_out == 4'd1 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm_out < 1'd1 & fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign add0_left =
     ~i0_done & cs_wh_out & fsm20_out == 4'd6 & go | fsm16_out == 2'd2 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go | fsm0_out == 3'd3 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? i0_out : 7'd0;
    assign add0_right =
     ~i0_done & cs_wh_out & fsm20_out == 4'd6 & go | fsm16_out == 2'd2 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go | fsm0_out == 3'd3 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? const4_out : 7'd0;
    assign add1_left =
     fsm3_out == 2'd1 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? i1_out : 7'd0;
    assign add1_right =
     fsm3_out == 2'd1 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? const8_out : 7'd0;
    assign add2_left =
     fsm13_out == 4'd0 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? a0_0_out :
     fsm12_out == 4'd12 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm12_out == 4'd1 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? doub_ixr_0_out : 64'd0;
    assign add2_right =
     fsm12_out == 4'd12 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? c_img_0_out :
     fsm12_out == 4'd1 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? c_real_0_out :
     fsm13_out == 4'd0 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? z_real_2_0_out : 64'd0;
    assign add5_left =
     fsm14_out == 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? n_iters0_out : 4'd0;
    assign add5_right =
     fsm14_out == 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? const15_out : 4'd0;
    assign bin_read0_0_clk =
     1'b1 ? clk : 1'd0;
    assign bin_read0_0_in =
     fsm7_out < 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? mult_pipe0_out : 64'd0;
    assign bin_read0_0_write_en =
     fsm7_out < 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? mult_pipe0_done : 1'd0;
    assign bin_read1_0_clk =
     1'b1 ? clk : 1'd0;
    assign bin_read1_0_in =
     fsm8_out < 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? mult_pipe1_out : 64'd0;
    assign bin_read1_0_write_en =
     fsm8_out < 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? mult_pipe1_done : 1'd0;
    assign bin_read2_0_clk =
     1'b1 ? clk : 1'd0;
    assign bin_read2_0_in =
     fsm10_out < 3'd4 & fsm11_out < 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? mult_pipe2_out : 64'd0;
    assign bin_read2_0_write_en =
     fsm10_out < 3'd4 & fsm11_out < 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? mult_pipe2_done : 1'd0;
    assign bin_read3_0_clk =
     1'b1 ? clk : 1'd0;
    assign bin_read3_0_in =
     fsm12_out >= 4'd7 & fsm12_out < 4'd11 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? mult_pipe3_out : 64'd0;
    assign bin_read3_0_write_en =
     fsm12_out >= 4'd7 & fsm12_out < 4'd11 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? mult_pipe3_done : 1'd0;
    assign c_img_0_clk =
     1'b1 ? clk : 1'd0;
    assign c_img_0_in =
     fsm5_out < 1'd1 & ~(fsm5_out == 1'd1) & cs_wh_out & fsm20_out == 4'd3 & go ? c_img_mem0_read_data :
     fsm0_out == 3'd0 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? int_c_real0_read_data : 64'd0;
    assign c_img_0_write_en =
     fsm0_out == 3'd0 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm5_out < 1'd1 & ~(fsm5_out == 1'd1) & cs_wh_out & fsm20_out == 4'd3 & go ? 1'd1 : 1'd0;
    assign c_img_mem0_addr0 =
     fsm5_out < 1'd1 & ~(fsm5_out == 1'd1) & cs_wh_out & fsm20_out == 4'd3 & go ? i0_out :
     fsm0_out == 3'd2 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? rsh0_out : 7'd0;
    assign c_img_mem0_clk =
     1'b1 ? clk : 1'd0;
    assign c_img_mem0_write_data =
     fsm0_out == 3'd2 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? a0_0_out : 64'd0;
    assign c_img_mem0_write_en =
     fsm0_out == 3'd2 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign c_real_0_clk =
     1'b1 ? clk : 1'd0;
    assign c_real_0_in =
     fsm5_out < 1'd1 & ~(fsm5_out == 1'd1) & cs_wh_out & fsm20_out == 4'd3 & go ? c_real_mem0_read_data : 64'd0;
    assign c_real_0_write_en =
     fsm5_out < 1'd1 & ~(fsm5_out == 1'd1) & cs_wh_out & fsm20_out == 4'd3 & go ? 1'd1 : 1'd0;
    assign c_real_mem0_addr0 =
     fsm5_out < 1'd1 & ~(fsm5_out == 1'd1) & cs_wh_out & fsm20_out == 4'd3 & go ? i0_out :
     fsm_out < 1'd1 & fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? rsh0_out : 7'd0;
    assign c_real_mem0_clk =
     1'b1 ? clk : 1'd0;
    assign c_real_mem0_write_data =
     fsm_out < 1'd1 & fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? c_img_0_out : 64'd0;
    assign c_real_mem0_write_en =
     fsm_out < 1'd1 & fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign cond_stored_clk =
     1'b1 ? clk : 1'd0;
    assign cond_stored_in =
     fsm1_out < 3'd1 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? le0_out : 1'd0;
    assign cond_stored_reset =
     1'b1 ? reset : 1'd0;
    assign cond_stored_write_en =
     fsm1_out < 3'd1 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign cond_stored0_clk =
     1'b1 ? clk : 1'd0;
    assign cond_stored0_in =
     fsm4_out < 2'd1 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? le1_out : 1'd0;
    assign cond_stored0_reset =
     1'b1 ? reset : 1'd0;
    assign cond_stored0_write_en =
     fsm4_out < 2'd1 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign cond_stored1_clk =
     1'b1 ? clk : 1'd0;
    assign cond_stored1_in =
     fsm13_out == 4'd0 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? or0_out : 1'd0;
    assign cond_stored1_reset =
     1'b1 ? reset : 1'd0;
    assign cond_stored1_write_en =
     fsm13_out == 4'd0 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign cond_stored2_clk =
     1'b1 ? clk : 1'd0;
    assign cond_stored2_in =
     fsm15_out < 5'd1 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? le3_out : 1'd0;
    assign cond_stored2_reset =
     1'b1 ? reset : 1'd0;
    assign cond_stored2_write_en =
     fsm15_out < 5'd1 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign cond_stored3_clk =
     1'b1 ? clk : 1'd0;
    assign cond_stored3_in =
     fsm17_out < 3'd1 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? le0_out : 1'd0;
    assign cond_stored3_reset =
     1'b1 ? reset : 1'd0;
    assign cond_stored3_write_en =
     fsm17_out < 3'd1 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? 1'd1 : 1'd0;
    assign cs_wh_clk =
     1'b1 ? clk : 1'd0;
    assign cs_wh_in =
     fsm20_out == 4'd8 & go ? 1'd0 :
     fsm20_out == 4'd2 & go ? le0_out : 1'd0;
    assign cs_wh_reset =
     1'b1 ? reset : 1'd0;
    assign cs_wh_write_en =
     fsm20_out == 4'd2 & go | fsm20_out == 4'd8 & go ? 1'd1 : 1'd0;
    assign doub_ixr_0_clk =
     1'b1 ? clk : 1'd0;
    assign doub_ixr_0_in =
     fsm10_out == 3'd4 & fsm11_out < 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? bin_read2_0_out :
     fsm12_out == 4'd11 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? bin_read3_0_out :
     fsm12_out == 4'd0 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? sub0_out : 64'd0;
    assign doub_ixr_0_write_en =
     fsm10_out == 3'd4 & fsm11_out < 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm12_out == 4'd11 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm12_out == 4'd0 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign eq0_left =
     fsm13_out == 4'd0 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? outputs_read0_0_out : 1'd0;
    assign eq0_right =
     fsm13_out == 4'd0 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? const13_out : 1'd0;
    assign fsm_clk =
     1'b1 ? clk : 1'd0;
    assign fsm_in =
     fsm_out == 1'd1 ? 1'd0 :
     fsm_out != 1'd1 & fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? incr_out : 1'd0;
    assign fsm_reset =
     1'b1 ? reset : 1'd0;
    assign fsm_write_en =
     fsm_out != 1'd1 & fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm_out == 1'd1 ? 1'd1 : 1'd0;
    assign fsm0_clk =
     1'b1 ? clk : 1'd0;
    assign fsm0_in =
     fsm0_out == 3'd4 ? 3'd0 :
     fsm0_out != 3'd4 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? incr0_out : 3'd0;
    assign fsm0_reset =
     1'b1 ? reset : 1'd0;
    assign fsm0_write_en =
     fsm0_out != 3'd4 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm0_out == 3'd4 ? 1'd1 : 1'd0;
    assign fsm1_clk =
     1'b1 ? clk : 1'd0;
    assign fsm1_in =
     fsm1_out == 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm1_out == 3'd1 & ~cond_stored_out ? 3'd0 :
     fsm1_out != 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? incr1_out : 3'd0;
    assign fsm1_reset =
     1'b1 ? reset : 1'd0;
    assign fsm1_write_en =
     fsm1_out != 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm1_out == 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm1_out == 3'd1 & ~cond_stored_out ? 1'd1 : 1'd0;
    assign fsm10_clk =
     1'b1 ? clk : 1'd0;
    assign fsm10_in =
     fsm10_out == 3'd5 ? 3'd0 :
     fsm10_out != 3'd5 & fsm11_out < 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? incr10_out : 3'd0;
    assign fsm10_reset =
     1'b1 ? reset : 1'd0;
    assign fsm10_write_en =
     fsm10_out != 3'd5 & fsm11_out < 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm10_out == 3'd5 ? 1'd1 : 1'd0;
    assign fsm11_clk =
     1'b1 ? clk : 1'd0;
    assign fsm11_in =
     fsm11_out == 3'd5 ? 3'd0 :
     fsm11_out != 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? incr11_out : 3'd0;
    assign fsm11_reset =
     1'b1 ? reset : 1'd0;
    assign fsm11_write_en =
     fsm11_out != 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm11_out == 3'd5 ? 1'd1 : 1'd0;
    assign fsm12_clk =
     1'b1 ? clk : 1'd0;
    assign fsm12_in =
     fsm12_out == 4'd14 ? 4'd0 :
     fsm12_out != 4'd14 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? incr12_out : 4'd0;
    assign fsm12_reset =
     1'b1 ? reset : 1'd0;
    assign fsm12_write_en =
     fsm12_out != 4'd14 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm12_out == 4'd14 ? 1'd1 : 1'd0;
    assign fsm13_clk =
     1'b1 ? clk : 1'd0;
    assign fsm13_in =
     fsm13_out == 4'd15 ? 4'd0 :
     fsm13_out != 4'd15 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? incr13_out : 4'd0;
    assign fsm13_reset =
     1'b1 ? reset : 1'd0;
    assign fsm13_write_en =
     fsm13_out != 4'd15 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm13_out == 4'd15 ? 1'd1 : 1'd0;
    assign fsm14_clk =
     1'b1 ? clk : 1'd0;
    assign fsm14_in =
     fsm14_out == 5'd22 ? 5'd0 :
     fsm14_out != 5'd22 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? incr14_out : 5'd0;
    assign fsm14_reset =
     1'b1 ? reset : 1'd0;
    assign fsm14_write_en =
     fsm14_out != 5'd22 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm14_out == 5'd22 ? 1'd1 : 1'd0;
    assign fsm15_clk =
     1'b1 ? clk : 1'd0;
    assign fsm15_in =
     fsm15_out == 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm15_out == 5'd1 & ~cond_stored2_out ? 5'd0 :
     fsm15_out != 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? incr15_out : 5'd0;
    assign fsm15_reset =
     1'b1 ? reset : 1'd0;
    assign fsm15_write_en =
     fsm15_out != 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm15_out == 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm15_out == 5'd1 & ~cond_stored2_out ? 1'd1 : 1'd0;
    assign fsm16_clk =
     1'b1 ? clk : 1'd0;
    assign fsm16_in =
     fsm16_out == 2'd3 ? 2'd0 :
     fsm16_out != 2'd3 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? incr16_out : 2'd0;
    assign fsm16_reset =
     1'b1 ? reset : 1'd0;
    assign fsm16_write_en =
     fsm16_out != 2'd3 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go | fsm16_out == 2'd3 ? 1'd1 : 1'd0;
    assign fsm17_clk =
     1'b1 ? clk : 1'd0;
    assign fsm17_in =
     fsm17_out == 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go | fsm17_out == 3'd1 & ~cond_stored3_out ? 3'd0 :
     fsm17_out != 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? incr17_out : 3'd0;
    assign fsm17_reset =
     1'b1 ? reset : 1'd0;
    assign fsm17_write_en =
     fsm17_out != 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go | fsm17_out == 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go | fsm17_out == 3'd1 & ~cond_stored3_out ? 1'd1 : 1'd0;
    assign fsm18_clk =
     1'b1 ? clk : 1'd0;
    assign fsm18_in =
     fsm18_out == 2'd2 ? 2'd0 :
     fsm18_out == 2'd0 & i0_done & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 2'd1 :
     fsm18_out == 2'd1 & fsm1_out == 3'd1 & ~cond_stored_out & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 2'd2 : 2'd0;
    assign fsm18_reset =
     1'b1 ? reset : 1'd0;
    assign fsm18_write_en =
     fsm18_out == 2'd0 & i0_done & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm18_out == 2'd1 & fsm1_out == 3'd1 & ~cond_stored_out & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm18_out == 2'd2 ? 1'd1 : 1'd0;
    assign fsm19_clk =
     1'b1 ? clk : 1'd0;
    assign fsm19_in =
     fsm19_out == 2'd2 ? 2'd0 :
     fsm19_out == 2'd0 & i1_done & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 2'd1 :
     fsm19_out == 2'd1 & fsm4_out == 2'd1 & ~cond_stored0_out & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 2'd2 : 2'd0;
    assign fsm19_reset =
     1'b1 ? reset : 1'd0;
    assign fsm19_write_en =
     fsm19_out == 2'd0 & i1_done & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm19_out == 2'd1 & fsm4_out == 2'd1 & ~cond_stored0_out & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm19_out == 2'd2 ? 1'd1 : 1'd0;
    assign fsm2_clk =
     1'b1 ? clk : 1'd0;
    assign fsm2_in =
     fsm2_out == 1'd1 ? 1'd0 :
     fsm2_out != 1'd1 & fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? incr2_out : 1'd0;
    assign fsm2_reset =
     1'b1 ? reset : 1'd0;
    assign fsm2_write_en =
     fsm2_out != 1'd1 & fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm2_out == 1'd1 ? 1'd1 : 1'd0;
    assign fsm20_clk =
     1'b1 ? clk : 1'd0;
    assign fsm20_in =
     fsm20_out == 4'd10 ? 4'd0 :
     fsm20_out == 4'd9 & fsm17_out == 3'd1 & ~cond_stored3_out & go ? 4'd10 :
     fsm20_out == 4'd0 & pd_out & pd0_out & go ? 4'd1 :
     fsm20_out == 4'd1 & i0_done & go | fsm20_out == 4'd7 & cs_wh_out & go ? 4'd2 :
     fsm20_out == 4'd2 & 1'b1 & go ? 4'd3 :
     fsm20_out == 4'd3 & fsm5_out == 1'd1 & cs_wh_out & go ? 4'd4 :
     fsm20_out == 4'd4 & n_iters0_done & cs_wh_out & go ? 4'd5 :
     fsm20_out == 4'd5 & fsm15_out == 5'd1 & ~cond_stored2_out & cs_wh_out & go ? 4'd6 :
     fsm20_out == 4'd6 & i0_done & cs_wh_out & go ? 4'd7 :
     fsm20_out == 4'd3 & ~cs_wh_out & go ? 4'd8 :
     fsm20_out == 4'd8 & i0_done & go ? 4'd9 : 4'd0;
    assign fsm20_reset =
     1'b1 ? reset : 1'd0;
    assign fsm20_write_en =
     fsm20_out == 4'd0 & pd_out & pd0_out & go | fsm20_out == 4'd1 & i0_done & go | fsm20_out == 4'd2 & 1'b1 & go | fsm20_out == 4'd3 & fsm5_out == 1'd1 & cs_wh_out & go | fsm20_out == 4'd4 & n_iters0_done & cs_wh_out & go | fsm20_out == 4'd5 & fsm15_out == 5'd1 & ~cond_stored2_out & cs_wh_out & go | fsm20_out == 4'd6 & i0_done & cs_wh_out & go | fsm20_out == 4'd7 & cs_wh_out & go | fsm20_out == 4'd3 & ~cs_wh_out & go | fsm20_out == 4'd8 & i0_done & go | fsm20_out == 4'd9 & fsm17_out == 3'd1 & ~cond_stored3_out & go | fsm20_out == 4'd10 ? 1'd1 : 1'd0;
    assign fsm3_clk =
     1'b1 ? clk : 1'd0;
    assign fsm3_in =
     fsm3_out == 2'd2 ? 2'd0 :
     fsm3_out != 2'd2 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? incr3_out : 2'd0;
    assign fsm3_reset =
     1'b1 ? reset : 1'd0;
    assign fsm3_write_en =
     fsm3_out != 2'd2 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm3_out == 2'd2 ? 1'd1 : 1'd0;
    assign fsm4_clk =
     1'b1 ? clk : 1'd0;
    assign fsm4_in =
     fsm4_out == 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm4_out == 2'd1 & ~cond_stored0_out ? 2'd0 :
     fsm4_out != 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? incr4_out : 2'd0;
    assign fsm4_reset =
     1'b1 ? reset : 1'd0;
    assign fsm4_write_en =
     fsm4_out != 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm4_out == 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm4_out == 2'd1 & ~cond_stored0_out ? 1'd1 : 1'd0;
    assign fsm5_clk =
     1'b1 ? clk : 1'd0;
    assign fsm5_in =
     fsm5_out == 1'd1 ? 1'd0 :
     fsm5_out != 1'd1 & ~(fsm5_out == 1'd1) & cs_wh_out & fsm20_out == 4'd3 & go ? incr5_out : 1'd0;
    assign fsm5_reset =
     1'b1 ? reset : 1'd0;
    assign fsm5_write_en =
     fsm5_out != 1'd1 & ~(fsm5_out == 1'd1) & cs_wh_out & fsm20_out == 4'd3 & go | fsm5_out == 1'd1 ? 1'd1 : 1'd0;
    assign fsm6_clk =
     1'b1 ? clk : 1'd0;
    assign fsm6_in =
     fsm6_out == 1'd1 ? 1'd0 :
     fsm6_out != 1'd1 & fsm14_out == 5'd0 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? incr6_out : 1'd0;
    assign fsm6_reset =
     1'b1 ? reset : 1'd0;
    assign fsm6_write_en =
     fsm6_out != 1'd1 & fsm14_out == 5'd0 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm6_out == 1'd1 ? 1'd1 : 1'd0;
    assign fsm7_clk =
     1'b1 ? clk : 1'd0;
    assign fsm7_in =
     fsm7_out == 3'd5 ? 3'd0 :
     fsm7_out != 3'd5 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? incr7_out : 3'd0;
    assign fsm7_reset =
     1'b1 ? reset : 1'd0;
    assign fsm7_write_en =
     fsm7_out != 3'd5 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm7_out == 3'd5 ? 1'd1 : 1'd0;
    assign fsm8_clk =
     1'b1 ? clk : 1'd0;
    assign fsm8_in =
     fsm8_out == 3'd5 ? 3'd0 :
     fsm8_out != 3'd5 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? incr8_out : 3'd0;
    assign fsm8_reset =
     1'b1 ? reset : 1'd0;
    assign fsm8_write_en =
     fsm8_out != 3'd5 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm8_out == 3'd5 ? 1'd1 : 1'd0;
    assign fsm9_clk =
     1'b1 ? clk : 1'd0;
    assign fsm9_in =
     fsm9_out == 3'd5 ? 3'd0 :
     fsm9_out != 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? incr9_out : 3'd0;
    assign fsm9_reset =
     1'b1 ? reset : 1'd0;
    assign fsm9_write_en =
     fsm9_out != 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm9_out == 3'd5 ? 1'd1 : 1'd0;
    assign gt0_left =
     fsm13_out == 4'd0 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? add2_out : 64'd0;
    assign gt0_right =
     fsm13_out == 4'd0 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fp_const2_out : 64'd0;
    assign i0_clk =
     1'b1 ? clk : 1'd0;
    assign i0_in =
     ~i0_done & cs_wh_out & fsm20_out == 4'd6 & go | fsm16_out == 2'd2 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go | fsm0_out == 3'd3 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? add0_out :
     ~i0_done & fsm18_out == 2'd0 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | ~i0_done & fsm20_out == 4'd8 & go | ~i0_done & fsm20_out == 4'd1 & go ? const0_out : 7'd0;
    assign i0_write_en =
     ~i0_done & fsm18_out == 2'd0 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | ~i0_done & fsm20_out == 4'd8 & go | ~i0_done & fsm20_out == 4'd1 & go | ~i0_done & cs_wh_out & fsm20_out == 4'd6 & go | fsm16_out == 2'd2 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go | fsm0_out == 3'd3 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign i1_clk =
     1'b1 ? clk : 1'd0;
    assign i1_in =
     fsm3_out == 2'd1 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? add1_out :
     ~i1_done & fsm19_out == 2'd0 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? const5_out : 7'd0;
    assign i1_write_en =
     ~i1_done & fsm19_out == 2'd0 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm3_out == 2'd1 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign incr_left =
     fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign incr_right =
     fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? fsm_out : 1'd0;
    assign incr0_left =
     cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 3'd1 : 3'd0;
    assign incr0_right =
     cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? fsm0_out : 3'd0;
    assign incr1_left =
     ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? fsm1_out : 3'd0;
    assign incr1_right =
     ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 3'd1 : 3'd0;
    assign incr10_left =
     fsm11_out < 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 3'd1 : 3'd0;
    assign incr10_right =
     fsm11_out < 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fsm10_out : 3'd0;
    assign incr11_left =
     fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 3'd1 : 3'd0;
    assign incr11_right =
     fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fsm11_out : 3'd0;
    assign incr12_left =
     fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 4'd1 : 4'd0;
    assign incr12_right =
     fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fsm12_out : 4'd0;
    assign incr13_left =
     fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fsm13_out : 4'd0;
    assign incr13_right =
     fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 4'd1 : 4'd0;
    assign incr14_left =
     cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 5'd1 : 5'd0;
    assign incr14_right =
     cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fsm14_out : 5'd0;
    assign incr15_left =
     ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fsm15_out : 5'd0;
    assign incr15_right =
     ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 5'd1 : 5'd0;
    assign incr16_left =
     cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? 2'd1 : 2'd0;
    assign incr16_right =
     cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? fsm16_out : 2'd0;
    assign incr17_left =
     ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? fsm17_out : 3'd0;
    assign incr17_right =
     ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? 3'd1 : 3'd0;
    assign incr2_left =
     fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign incr2_right =
     fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? fsm2_out : 1'd0;
    assign incr3_left =
     cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 2'd1 : 2'd0;
    assign incr3_right =
     cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? fsm3_out : 2'd0;
    assign incr4_left =
     ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? fsm4_out : 2'd0;
    assign incr4_right =
     ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 2'd1 : 2'd0;
    assign incr5_left =
     ~(fsm5_out == 1'd1) & cs_wh_out & fsm20_out == 4'd3 & go ? 1'd1 : 1'd0;
    assign incr5_right =
     ~(fsm5_out == 1'd1) & cs_wh_out & fsm20_out == 4'd3 & go ? fsm5_out : 1'd0;
    assign incr6_left =
     fsm14_out == 5'd0 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign incr6_right =
     fsm14_out == 5'd0 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fsm6_out : 1'd0;
    assign incr7_left =
     fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 3'd1 : 3'd0;
    assign incr7_right =
     fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fsm7_out : 3'd0;
    assign incr8_left =
     fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 3'd1 : 3'd0;
    assign incr8_right =
     fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fsm8_out : 3'd0;
    assign incr9_left =
     fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 3'd1 : 3'd0;
    assign incr9_right =
     fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fsm9_out : 3'd0;
    assign int_c_img0_addr0 =
     fsm_out < 1'd1 & fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? i0_out : 7'd0;
    assign int_c_img0_clk =
     1'b1 ? clk : 1'd0;
    assign int_c_real0_addr0 =
     fsm0_out == 3'd0 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? i0_out : 7'd0;
    assign int_c_real0_clk =
     1'b1 ? clk : 1'd0;
    assign int_outputs0_addr0 =
     fsm16_out == 2'd1 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? i0_out : 7'd0;
    assign int_outputs0_clk =
     1'b1 ? clk : 1'd0;
    assign int_outputs0_write_data =
     fsm16_out == 2'd1 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? outputs_read0_0_out : 1'd0;
    assign int_outputs0_write_en =
     fsm16_out == 2'd1 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? 1'd1 : 1'd0;
    assign le0_left =
     fsm1_out < 3'd1 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm20_out == 4'd2 & go | fsm17_out < 3'd1 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? i0_out : 7'd0;
    assign le0_right =
     fsm1_out < 3'd1 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm20_out == 4'd2 & go | fsm17_out < 3'd1 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? const1_out : 7'd0;
    assign le1_left =
     fsm4_out < 2'd1 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? i1_out : 7'd0;
    assign le1_right =
     fsm4_out < 2'd1 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? const6_out : 7'd0;
    assign le3_left =
     fsm15_out < 5'd1 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? n_iters0_out : 4'd0;
    assign le3_right =
     fsm15_out < 5'd1 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? const12_out : 4'd0;
    assign mult_pipe0_clk =
     1'b1 ? clk : 1'd0;
    assign mult_pipe0_go =
     ~mult_pipe0_done & fsm7_out < 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign mult_pipe0_left =
     fsm7_out < 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? z_img_0_out : 64'd0;
    assign mult_pipe0_right =
     fsm7_out < 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? z_img_0_out : 64'd0;
    assign mult_pipe1_clk =
     1'b1 ? clk : 1'd0;
    assign mult_pipe1_go =
     ~mult_pipe1_done & fsm8_out < 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign mult_pipe1_left =
     fsm8_out < 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? z_real_0_out : 64'd0;
    assign mult_pipe1_right =
     fsm8_out < 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? z_real_0_out : 64'd0;
    assign mult_pipe2_clk =
     1'b1 ? clk : 1'd0;
    assign mult_pipe2_go =
     ~mult_pipe2_done & fsm10_out < 3'd4 & fsm11_out < 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign mult_pipe2_left =
     fsm10_out < 3'd4 & fsm11_out < 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? z_img_0_out : 64'd0;
    assign mult_pipe2_right =
     fsm10_out < 3'd4 & fsm11_out < 3'd5 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? z_real_0_out : 64'd0;
    assign mult_pipe3_clk =
     1'b1 ? clk : 1'd0;
    assign mult_pipe3_go =
     ~mult_pipe3_done & fsm12_out >= 4'd7 & fsm12_out < 4'd11 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign mult_pipe3_left =
     fsm12_out >= 4'd7 & fsm12_out < 4'd11 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? fp_const3_out : 64'd0;
    assign mult_pipe3_right =
     fsm12_out >= 4'd7 & fsm12_out < 4'd11 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? doub_ixr_0_out : 64'd0;
    assign n_iters0_clk =
     1'b1 ? clk : 1'd0;
    assign n_iters0_in =
     fsm14_out == 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? add5_out :
     ~n_iters0_done & cs_wh_out & fsm20_out == 4'd4 & go ? const11_out : 4'd0;
    assign n_iters0_write_en =
     ~n_iters0_done & cs_wh_out & fsm20_out == 4'd4 & go | fsm14_out == 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign or0_left =
     fsm13_out == 4'd0 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? eq0_out : 1'd0;
    assign or0_right =
     fsm13_out == 4'd0 & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? gt0_out : 1'd0;
    assign outputs0_addr0 =
     fsm9_out < 3'd1 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm13_out > 4'd0 & fsm13_out < 4'd2 & cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? i0_out :
     fsm2_out < 1'd1 & fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? i1_out :
     fsm16_out == 2'd0 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? rsh0_out : 7'd0;
    assign outputs0_clk =
     1'b1 ? clk : 1'd0;
    assign outputs0_write_data =
     fsm13_out > 4'd0 & fsm13_out < 4'd2 & cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? const13_out :
     fsm2_out < 1'd1 & fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? const7_out : 1'd0;
    assign outputs0_write_en =
     fsm13_out > 4'd0 & fsm13_out < 4'd2 & cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm2_out < 1'd1 & fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign outputs_read0_0_clk =
     1'b1 ? clk : 1'd0;
    assign outputs_read0_0_in =
     fsm9_out < 3'd1 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm16_out == 2'd0 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? outputs0_read_data : 1'd0;
    assign outputs_read0_0_write_en =
     fsm9_out < 3'd1 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm16_out == 2'd0 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go ? 1'd1 : 1'd0;
    assign pd_clk =
     1'b1 ? clk : 1'd0;
    assign pd_in =
     pd_out & pd0_out ? 1'd0 :
     fsm18_out == 2'd2 & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign pd_reset =
     1'b1 ? reset : 1'd0;
    assign pd_write_en =
     fsm18_out == 2'd2 & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | pd_out & pd0_out ? 1'd1 : 1'd0;
    assign pd0_clk =
     1'b1 ? clk : 1'd0;
    assign pd0_in =
     pd_out & pd0_out ? 1'd0 :
     fsm19_out == 2'd2 & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign pd0_reset =
     1'b1 ? reset : 1'd0;
    assign pd0_write_en =
     fsm19_out == 2'd2 & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | pd_out & pd0_out ? 1'd1 : 1'd0;
    assign rsh0_left =
     fsm_out < 1'd1 & fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm16_out == 2'd0 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go | fsm0_out == 3'd2 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? i0_out : 7'd0;
    assign rsh0_right =
     fsm_out < 1'd1 & fsm0_out == 3'd1 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go | fsm16_out == 2'd0 & cond_stored3_out & fsm17_out >= 3'd1 & fsm17_out < 3'd4 & ~(fsm17_out == 3'd1 & ~cond_stored3_out) & fsm20_out == 4'd9 & go | fsm0_out == 3'd2 & cond_stored_out & fsm1_out >= 3'd1 & fsm1_out < 3'd5 & ~(fsm1_out == 3'd1 & ~cond_stored_out) & fsm18_out == 2'd1 & ~(pd_out | fsm18_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? const0_out : 7'd0;
    assign sub0_left =
     fsm12_out == 4'd0 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? z_real_2_0_out : 64'd0;
    assign sub0_right =
     fsm12_out == 4'd0 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? a0_0_out : 64'd0;
    assign z_img_0_clk =
     1'b1 ? clk : 1'd0;
    assign z_img_0_in =
     fsm6_out < 1'd1 & fsm14_out == 5'd0 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? z_img_mem0_read_data : 64'd0;
    assign z_img_0_write_en =
     fsm6_out < 1'd1 & fsm14_out == 5'd0 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign z_img_mem0_addr0 =
     fsm6_out < 1'd1 & fsm14_out == 5'd0 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm12_out == 4'd13 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? i0_out :
     fsm2_out < 1'd1 & fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? i1_out : 7'd0;
    assign z_img_mem0_clk =
     1'b1 ? clk : 1'd0;
    assign z_img_mem0_write_data =
     fsm12_out == 4'd13 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? a0_0_out :
     fsm2_out < 1'd1 & fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? fp_const0_out : 64'd0;
    assign z_img_mem0_write_en =
     fsm12_out == 4'd13 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm2_out < 1'd1 & fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
    assign z_real_0_clk =
     1'b1 ? clk : 1'd0;
    assign z_real_0_in =
     fsm6_out < 1'd1 & fsm14_out == 5'd0 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? z_real_mem0_read_data : 64'd0;
    assign z_real_0_write_en =
     fsm6_out < 1'd1 & fsm14_out == 5'd0 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign z_real_2_0_clk =
     1'b1 ? clk : 1'd0;
    assign z_real_2_0_in =
     fsm8_out == 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? bin_read1_0_out : 64'd0;
    assign z_real_2_0_write_en =
     fsm8_out == 3'd4 & fsm9_out < 3'd5 & fsm14_out >= 5'd1 & fsm14_out < 5'd6 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? 1'd1 : 1'd0;
    assign z_real_mem0_addr0 =
     fsm6_out < 1'd1 & fsm14_out == 5'd0 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm11_out < 3'd1 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? i0_out :
     fsm2_out < 1'd1 & fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? i1_out : 7'd0;
    assign z_real_mem0_clk =
     1'b1 ? clk : 1'd0;
    assign z_real_mem0_write_data =
     fsm11_out < 3'd1 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go ? a0_0_out :
     fsm2_out < 1'd1 & fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? fp_const1_out : 64'd0;
    assign z_real_mem0_write_en =
     fsm11_out < 3'd1 & fsm12_out >= 4'd2 & fsm12_out < 4'd7 & fsm13_out > 4'd0 & fsm13_out < 4'd15 & ~cond_stored1_out & fsm14_out >= 5'd6 & fsm14_out < 5'd21 & cond_stored2_out & fsm15_out >= 5'd1 & fsm15_out < 5'd23 & ~(fsm15_out == 5'd1 & ~cond_stored2_out) & cs_wh_out & fsm20_out == 4'd5 & go | fsm2_out < 1'd1 & fsm3_out == 2'd0 & cond_stored0_out & fsm4_out >= 2'd1 & fsm4_out < 2'd3 & ~(fsm4_out == 2'd1 & ~cond_stored0_out) & fsm19_out == 2'd1 & ~(pd0_out | fsm19_out == 2'd2) & ~(pd_out & pd0_out) & fsm20_out == 4'd0 & go ? 1'd1 : 1'd0;
endmodule

