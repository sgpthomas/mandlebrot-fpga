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
